"""
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
Selection of black-and-white reference frame candidates by DINOv3 semantic
similarity.

Removes near-duplicate scenes among the candidates produced by a scene
detector (e.g. SceneDetectEdges) before they are sent out for colorization
(Qwen or similar): a long shot-reverse-shot dialogue can otherwise yield many
near-identical candidates of the same location, and when ColorMNet's
permanent memory holds several slightly different references of the same
scene, its softmax-weighted readout tends to average them into desaturated
colors.

Two public entry points (both re-exported by vscmnet2/__init__.py), sharing
the same underlying work via the private _SelectionRun driver:

  select_reference_frames()    plain Python, returns the result dict
                                immediately. Not VapourSynth-friendly (no
                                clip in, no clip out) - for direct/CLI use.
  vs_select_reference_frames()  VapourSynth-friendly, styled like
                                vs_sc_export_frames(): returns a placeholder
                                clip and does its work lazily, one candidate
                                at a time as frames are requested, instead of
                                blocking for the whole run at script-eval
                                time (which is what calling
                                select_reference_frames() directly at a
                                .vpy's top level would do).
"""

from __future__ import annotations

import base64
import io
import json
import os
from collections import defaultdict
from typing import Optional

import numpy as np
from PIL import Image
import vapoursynth as vs

from .vsslib.vsutils import get_ref_names, get_ref_num, CMNET2_LogMessage, MessageType
from .vsslib.models_config import get_cmnet2_model, check_file

package_dir = os.path.dirname(os.path.realpath(__file__))

# ImageNet normalisation statistics, applied to the DINOv3 ViT-B/16 input.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Shared validation / discovery
# ---------------------------------------------------------------------------

def _validate_and_discover(ref_framedir: str, out_framedir: str, similarity_threshold: float,
                            select_window: int, input_size: int) -> tuple:
    """Cheap parameter validation + candidate discovery, shared by
    select_reference_frames() and vs_select_reference_frames(). No GPU/model
    work and no image decoding here - safe to run eagerly, even from the
    VapourSynth-lazy entry point (matches vs_export_reference_frames(),
    which also validates/creates directories eagerly before returning its
    lazy clip). Raises via CMNET2_LogMessage(EXCEPTION, ...) on any invalid
    input. Returns (frame_ids, filenames), both sorted ascending by frame_id.
    """
    if not os.path.isdir(ref_framedir):
        CMNET2_LogMessage(MessageType.EXCEPTION,
                           "select_reference_frames: input directory not found:", ref_framedir)

    candidate_names = get_ref_names(ref_framedir)
    if not candidate_names:
        CMNET2_LogMessage(MessageType.EXCEPTION,
                           "select_reference_frames: no reference frames (ref_NNNNNN.ext) found in",
                           ref_framedir)

    if not (0.0 < similarity_threshold <= 1.0):
        CMNET2_LogMessage(MessageType.EXCEPTION,
                           "select_reference_frames: similarity_threshold must be in (0, 1], got",
                           similarity_threshold)

    if select_window < 0:
        CMNET2_LogMessage(MessageType.EXCEPTION,
                           "select_reference_frames: select_window must be >= 0, got", select_window)

    if input_size % 16 != 0:
        CMNET2_LogMessage(MessageType.EXCEPTION,
                           "select_reference_frames: input_size must be a multiple of 16"
                           " (DINOv3 ViT-B/16 patch size), got", input_size)

    ref_resolved = os.path.normcase(os.path.realpath(ref_framedir))
    out_resolved = os.path.normcase(os.path.realpath(out_framedir))
    if ref_resolved == out_resolved:
        CMNET2_LogMessage(MessageType.EXCEPTION,
                           "select_reference_frames: out_framedir must differ from ref_framedir, both resolve to",
                           ref_resolved)

    # out_framedir already holding candidates from a previous run: refuse to
    # overwrite/delete. Checked here, before any GPU/model work or image
    # decoding, and unconditionally (including dry_run=True, which still
    # writes cluster_map.json into out_framedir further down). A missing
    # out_framedir is not an error - it has no pre-existing ref_* by
    # definition, and is created later in the Output step.
    if os.path.isdir(out_framedir):
        existing_out = get_ref_names(out_framedir)
        if existing_out:
            CMNET2_LogMessage(MessageType.EXCEPTION,
                               f"select_reference_frames: out_framedir already contains"
                               f" {len(existing_out)} reference frame(s), refusing to overwrite:", out_framedir)

    # sort candidates by frame_id, and refuse silently-colliding frame ids
    pairs = []
    seen_ids = {}
    for name in candidate_names:
        fid = get_ref_num(name)
        if fid in seen_ids:
            CMNET2_LogMessage(MessageType.EXCEPTION,
                               f"select_reference_frames: duplicate frame id {fid} in {ref_framedir}:",
                               seen_ids[fid], "and", name)
        seen_ids[fid] = name
        pairs.append((fid, name))
    pairs.sort(key=lambda t: t[0])
    frame_ids = [p[0] for p in pairs]
    filenames = [p[1] for p in pairs]

    CMNET2_LogMessage(MessageType.INFORMATION,
                       f"select_reference_frames: {len(pairs)} candidates in {ref_framedir}"
                       f" (similarity_threshold={similarity_threshold}, select_window={select_window},"
                       f" input_size={input_size})")

    return frame_ids, filenames


# ---------------------------------------------------------------------------
# Shared run driver
# ---------------------------------------------------------------------------

class _SelectionRun:
    """Stateful driver for one selection run: owns the DINOv3 feature
    extractor and the streaming feature/mean-L/thumbnail buffers, and
    performs the final clustering + Output step. Shared by
    select_reference_frames() (every candidate fed in with one synchronous
    call) and vs_select_reference_frames() (one candidate fed in per
    VapourSynth frame request) so the actual work is implemented exactly
    once. Not part of the public API of this module.
    """

    def __init__(self, ref_framedir: str, out_framedir: str, similarity_threshold: float,
                 select_window: int, input_size: int, batch_size: int, device: str,
                 dry_run: bool, debug_html: bool, move_files: bool, frame_ids: list, filenames: list):
        self.ref_framedir = ref_framedir
        self.out_framedir = out_framedir
        self.similarity_threshold = similarity_threshold
        self.select_window = select_window
        self.input_size = input_size
        self.batch_size = batch_size
        self.device = device
        self.dry_run = dry_run
        self.debug_html = debug_html
        self.move_files = move_files
        self.frame_ids = frame_ids
        self.filenames = filenames
        self.n = len(frame_ids)

        self._extractor: Optional["_DinoV3FeatureExtractor"] = None
        self.features: Optional[np.ndarray] = None
        self.mean_l = np.empty((self.n,), dtype=np.float32)
        self.thumbs_b64 = [None] * self.n if debug_html else None

        self._pending_paths: list = []
        self._pending_idx: list = []
        num_batches = (self.n + batch_size - 1) // batch_size
        self._log_every = max(1, num_batches // 10)
        self._batches_done = 0
        self._done_count = 0

    def add_image(self, idx: int) -> None:
        """Buffer candidate at sorted position `idx`. Loads the backbone on
        the first call. Flushes (one backbone forward pass over the pending
        batch) once batch_size candidates are pending, or `idx` is the last
        one - so feeding candidates one at a time (as
        vs_select_reference_frames() does, one per VapourSynth frame) still
        runs the backbone in batches, same as select_reference_frames()'s
        single synchronous call.
        """
        if self._extractor is None:
            self._extractor = _DinoV3FeatureExtractor(device=self.device)
        self._pending_paths.append(os.path.join(self.ref_framedir, self.filenames[idx]))
        self._pending_idx.append(idx)
        if len(self._pending_paths) >= self.batch_size or idx == self.n - 1:
            self._flush()

    def _flush(self) -> None:
        if not self._pending_paths:
            return
        pil_batch = [Image.open(p).convert("RGB") for p in self._pending_paths]
        feats = self._extractor.extract(pil_batch, self.input_size)
        if self.features is None:
            self.features = np.empty((self.n, feats.shape[1]), dtype=np.float32)

        for j, img in enumerate(pil_batch):
            idx = self._pending_idx[j]
            self.features[idx] = feats[j]
            self.mean_l[idx] = _mean_l_channel(img)
            if self.debug_html:
                self.thumbs_b64[idx] = _thumbnail_base64(img)
        self._done_count += len(pil_batch)
        self._batches_done += 1

        if self._batches_done % self._log_every == 0 or self._done_count == self.n:
            CMNET2_LogMessage(MessageType.INFORMATION,
                               f"select_reference_frames: extracted features for"
                               f" {self._done_count}/{self.n} candidates")
        self._pending_paths = []
        self._pending_idx = []

    def finalize(self) -> dict:
        """Cluster the accumulated features and write cluster_map.json,
        copy or move the representatives (unless dry_run) and
        cluster_debug.html (if debug_html). Call exactly once, after every
        candidate has been passed to add_image(). Returns the same dict
        documented by select_reference_frames()'s :return:.
        """
        if self._extractor is not None:
            self._extractor.release()
            self._extractor = None

        half = (self.select_window // 2) if self.select_window > 0 else (self.n - 1)
        labels = _cluster_banded(self.features, self.similarity_threshold, half)
        clusters = _build_clusters(self.frame_ids, self.filenames, labels, self.mean_l)

        os.makedirs(self.out_framedir, exist_ok=True)

        if not self.dry_run:
            # the "already contains ref_*" guard already ran in
            # _validate_and_discover(), before any GPU/model work.
            # Only the representatives are transferred - candidates that
            # lost their cluster are left untouched in ref_framedir, in
            # both copy and move mode.
            for c in clusters:
                src = os.path.join(self.ref_framedir, c["ref_file"])
                dst = os.path.join(self.out_framedir, c["ref_file"])
                _transfer_file(src, dst, self.move_files)

        cluster_map = {
            "params": {
                "similarity_threshold": self.similarity_threshold,
                "select_window": self.select_window,
                "input_size": self.input_size,
                "backbone": "dinov3",
            },
            "n_candidates": self.n,
            "clusters": [
                {
                    "ref_file": c["ref_file"],
                    "representative_frame_id": c["representative_frame_id"],
                    "member_frame_ids": c["member_frame_ids"],
                }
                for c in clusters
            ],
        }
        cluster_map_path = os.path.join(self.out_framedir, "cluster_map.json")
        with open(cluster_map_path, "w", encoding="utf-8") as f:
            json.dump(cluster_map, f, indent=2)

        debug_html_path = (
            _write_debug_html(clusters, self.frame_ids, self.thumbs_b64, self.out_framedir)
            if self.debug_html else None
        )

        n_clusters = len(clusters)
        reduction = 1.0 - n_clusters / self.n if self.n > 0 else 0.0
        cluster_sizes = [len(c["member_frame_ids"]) for c in clusters]

        CMNET2_LogMessage(MessageType.INFORMATION,
                           f"select_reference_frames: {self.n} candidates -> {n_clusters} clusters"
                           f" ({reduction * 100:.1f}% reduction), out_framedir={self.out_framedir}")

        return {
            "n_candidates": self.n,
            "n_clusters": n_clusters,
            "reduction_ratio": round(reduction, 4),
            "cluster_sizes": cluster_sizes,
            "out_framedir": str(self.out_framedir),
            "cluster_map": cluster_map_path,
            "debug_html": debug_html_path,
            "select_window": self.select_window,
            "similarity_threshold": self.similarity_threshold,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_reference_frames(ref_framedir: str, out_framedir: str, similarity_threshold: float = 0.95,
                             select_window: int = 50, input_size: int = 224, batch_size: int = 32,
                             device: str = "cuda", dry_run: bool = False, debug_html: bool = False,
                             move_files: bool = False) -> dict:
    """Select semantically distinct reference frame candidates with DINOv3.

    Clusters the candidate reference frames found in ref_framedir (files
    named ref_NNNNNN.ext, see get_ref_names/get_ref_num in vsslib/vsutils) by
    DINOv3 ViT-B/16 patch-feature cosine similarity, and copies (or moves,
    see move_files) only one representative per cluster to out_framedir,
    together with a cluster_map.json describing every original frame_id and
    which representative it maps to. out_framedir is meant to be used
    afterwards as the sc_framedir passed to the colorization filter, once the
    surviving representatives have been colorized externally (e.g. by Qwen).
    Near-duplicate candidates that lose the cluster are left untouched in
    ref_framedir either way: this function performs no expansion/hardlinking
    back to every original frame_id, out_framedir only ever contains the
    representatives.

    Plain Python, not VapourSynth-friendly (no clip in, no clip out): runs
    everything synchronously and returns the result dict immediately. Calling
    this directly at the top level of a .vpy script blocks the whole script
    evaluation for the full duration of the run - use
    vs_select_reference_frames() instead from VapourSynth hosts (vsedit,
    vspipe) that must stay responsive while it works.
    :param ref_framedir:         Directory with candidate reference frames,
                                  named ref_NNNNNN.ext. Never modified unless
                                  move_files=True, in which case the
                                  representative files (only those) are
                                  removed from here once moved to
                                  out_framedir.
    :param out_framedir:         Destination directory for the cluster
                                  representatives and cluster_map.json.
                                  Created if missing. Must resolve to a
                                  different path than ref_framedir.
    :param similarity_threshold: Cosine similarity at/above which two
                                  candidates are merged into the same
                                  cluster. Range (0, 1]. Default 0.95 is a
                                  starting point, not a calibrated value -
                                  recalibrate on a real clip (see the
                                  consecutive-candidate similarity
                                  percentiles reported by this function's
                                  verification script) before relying on it.
    :param select_window:        Band width expressed in CANDIDATE
                                  positions (index in the frame_id-sorted
                                  candidate list), NOT video frame numbers.
                                  Two candidates at list positions i, j are
                                  only compared (and can only be merged) if
                                  abs(i - j) <= select_window // 2.
                                  select_window=0 means global clustering
                                  (every candidate compared against every
                                  other candidate, regardless of distance).
                                  Deliberately independent of
                                  max_memory_frames (colormnet2_render.py):
                                  there is no derived default, no
                                  consistency check and no warning if the
                                  two disagree - a caller that wants them
                                  aligned must pass a matching value
                                  explicitly. An odd select_window loses
                                  half a candidate to the integer division
                                  (select_window // 2).
    :param input_size:           Square resolution fed to the DINOv3
                                  ViT-B/16 backbone. Must be a multiple of
                                  16 (its patch size). Default 224.
    :param batch_size:           Number of candidate images per backbone
                                  forward pass. Default 32.
    :param device:               "cuda" or "cpu". Falls back to "cpu" if
                                  CUDA is requested but not available.
                                  Default "cuda".
    :param dry_run:              If True, write cluster_map.json without
                                  copying any representative file to
                                  out_framedir. Default False.
    :param debug_html:           If True, also write cluster_debug.html to
                                  out_framedir: one section per multi-member
                                  cluster, with base64-encoded thumbnails of
                                  every member and the representative
                                  highlighted. Single-member clusters (no
                                  merge decision to review) are omitted from
                                  the page - they are still kept, unchanged,
                                  in cluster_map.json and out_framedir.
                                  Default False.
    :param move_files:           If True, representative files are moved
                                  (shutil.move) from ref_framedir to
                                  out_framedir instead of copied
                                  (shutil.copy2) - ref_framedir loses those
                                  files. Candidates that lost their cluster
                                  are left in ref_framedir either way.
                                  Ignored when dry_run=True (nothing is
                                  transferred). Default False (copy, matches
                                  every previously-verified run of this
                                  function).
    :return:                     dict with keys n_candidates, n_clusters,
                                  reduction_ratio, cluster_sizes,
                                  out_framedir, cluster_map (path to the
                                  written cluster_map.json), debug_html
                                  (path to the written cluster_debug.html
                                  when debug_html=True, else None),
                                  select_window, similarity_threshold.
    """
    frame_ids, filenames = _validate_and_discover(
        ref_framedir, out_framedir, similarity_threshold, select_window, input_size)
    run = _SelectionRun(ref_framedir, out_framedir, similarity_threshold, select_window,
                         input_size, batch_size, device, dry_run, debug_html, move_files,
                         frame_ids, filenames)
    for idx in range(run.n):
        run.add_image(idx)
    return run.finalize()


def vs_select_reference_frames(ref_framedir: str, out_framedir: str, similarity_threshold: float = 0.95,
                                select_window: int = 50, input_size: int = 224, batch_size: int = 32,
                                device: str = "cuda", dry_run: bool = False, debug_html: bool = False,
                                move_files: bool = False) -> vs.VideoNode:
    """VapourSynth-friendly wrapper for select_reference_frames().

    Same DINOv3-based semantic deduplication of reference frame candidates
    (see select_reference_frames() for the full mechanism), but styled like
    vs_sc_export_frames(): instead of blocking and returning a result dict,
    it returns a placeholder clip immediately and does the real work lazily,
    one candidate at a time, only as frames of that clip are requested -
    the backbone is still run in batches internally (batch_size), it is only
    the driving loop that becomes one candidate per VapourSynth frame. This
    keeps a VapourSynth host (vsedit, vspipe) responsive instead of freezing
    for the whole run during script evaluation, which is what happens if
    select_reference_frames() is called directly at a .vpy's top level.
    Validation (and directory creation on failure paths) still happens
    eagerly, at call time - only the GPU/model work and the final
    clustering + Output step are deferred.
    :param ref_framedir:         Directory with candidate reference frames,
                                  named ref_NNNNNN.ext. Never modified unless
                                  move_files=True, in which case the
                                  representative files (only those) are
                                  removed from here once moved to
                                  out_framedir.
    :param out_framedir:         Destination directory for the cluster
                                  representatives and cluster_map.json.
                                  Created if missing. Must resolve to a
                                  different path than ref_framedir.
    :param similarity_threshold: Cosine similarity at/above which two
                                  candidates are merged into the same
                                  cluster. Range (0, 1]. Default 0.95 is a
                                  starting point, not a calibrated value -
                                  recalibrate on a real clip before relying
                                  on it (see select_reference_frames()).
    :param select_window:        Band width in CANDIDATE positions, not video
                                  frame numbers; select_window=0 means global
                                  clustering. See select_reference_frames()
                                  for the exact semantics.
    :param input_size:           Square resolution fed to the DINOv3
                                  ViT-B/16 backbone. Must be a multiple of
                                  16 (its patch size). Default 224.
    :param batch_size:           Number of candidate images per backbone
                                  forward pass. Default 32.
    :param device:               "cuda" or "cpu". Falls back to "cpu" if
                                  CUDA is requested but not available.
                                  Default "cuda".
    :param dry_run:              If True, write cluster_map.json without
                                  copying any representative file to
                                  out_framedir. Default False.
    :param debug_html:           If True, also write cluster_debug.html to
                                  out_framedir. Default False.
    :param move_files:           If True, representative files are moved
                                  from ref_framedir to out_framedir instead
                                  of copied - see select_reference_frames().
                                  Ignored when dry_run=True. Default False.
    :return:                     A placeholder clip (BlankClip) whose length
                                  equals the number of candidates to process
                                  - it carries no useful pixel data, only a
                                  frame count and a hook for the deferred
                                  work. The result dict returned by
                                  select_reference_frames() is NOT exposed
                                  here: every useful output already lives in
                                  cluster_map.json / cluster_debug.html
                                  inside out_framedir once the last frame has
                                  been requested (e.g. by playing/exporting
                                  the whole clip, not just previewing frame 0).
    """
    frame_ids, filenames = _validate_and_discover(
        ref_framedir, out_framedir, similarity_threshold, select_window, input_size)
    run = _SelectionRun(ref_framedir, out_framedir, similarity_threshold, select_window,
                         input_size, batch_size, device, dry_run, debug_html, move_files,
                         frame_ids, filenames)

    def _step(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        run.add_image(n)
        if n == run.n - 1:
            run.finalize()
        return f.copy()

    placeholder = vs.core.std.BlankClip(width=64, height=64, length=run.n, fpsnum=1, fpsden=1)
    return placeholder.std.ModifyFrame(clips=[placeholder], selector=_step)


# ---------------------------------------------------------------------------
# DINOv3 feature extraction
# ---------------------------------------------------------------------------

class _DinoV3FeatureExtractor:
    """Loads the frozen, native DINOv3 ViT-B/16 backbone (same weights as
    ColorMNetRender2) and extracts mean-pooled, L2-normalised patch
    features. Not part of the public API of this module.
    """

    def __init__(self, device: str = "cuda", weights_dir: Optional[str] = None):
        import torch  # heavy import kept local, as in ColorMNetRender2

        if weights_dir is None:
            model_info = get_cmnet2_model("dinov3")
            weights_dir = os.path.join(package_dir, "weights", model_info["weights_dir"])

        check_file(os.path.join(weights_dir, "config.json"),
                   "DINOv3 backbone config (reference selection)")
        check_file(os.path.join(weights_dir, "model.safetensors"),
                   "DINOv3 backbone weights (reference selection)")

        with open(os.path.join(weights_dir, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
        # never hardcoded: DINOv3 ViT-B/16 ships 4 register tokens today, but
        # the token layout ([cls, register_0..N, patch_0..M]) depends on it
        self._num_register_tokens = int(config["num_register_tokens"])

        resolved_device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.device = torch.device(resolved_device)

        from .colormnet2.model.dinov3_vit import DINOv3ViT
        model = DINOv3ViT.from_pretrained_dir(weights_dir, map_location="cpu")
        model.eval()
        model.requires_grad_(False)
        model.to(self.device)
        self._model = model

        self._torch = torch
        self._transform_cache = {}

    def _build_transform(self, input_size: int):
        transform = self._transform_cache.get(input_size)
        if transform is None:
            from torchvision import transforms
            from torchvision.transforms import InterpolationMode
            transform = transforms.Compose([
                transforms.Resize(input_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ])
            self._transform_cache[input_size] = transform
        return transform

    def extract(self, images: list, input_size: int) -> np.ndarray:
        """Extract L2-normalised, mean-pooled patch features for a batch of
        RGB PIL images. Returns a float32 (N, hidden_size) array."""
        torch = self._torch
        transform = self._build_transform(input_size)
        tensors = [transform(img) for img in images]
        batch = torch.stack(tensors, dim=0).to(self.device)

        with torch.inference_mode():
            out = self._model(batch, output_hidden_states=False)
            # skip [cls, register_0..N]; last_hidden_state is already post-norm
            patches = out.last_hidden_state[:, 1 + self._num_register_tokens:, :]
            pooled = patches.mean(dim=1)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)

        return pooled.to(device="cpu", dtype=torch.float32).numpy()

    def release(self) -> None:
        """Free the backbone. Called once feature extraction is done."""
        if hasattr(self, "_model"):
            del self._model
        if self.device.type == "cuda":
            self._torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Clustering (no scipy dependency)
# ---------------------------------------------------------------------------

def _pick_block_rows(n: int, half: int, budget_bytes: int = 256 * 1024 * 1024, bytes_per_elem: int = 4) -> int:
    """Largest row-block size whose similarity block (block_rows x column_span)
    stays within budget_bytes, bounding memory even in global mode (half == n - 1)."""
    block_rows = 1024
    while block_rows > 1:
        column_span = min(n, block_rows + half)
        if block_rows * column_span * bytes_per_elem <= budget_bytes:
            break
        block_rows //= 2
    return max(block_rows, 1)


def _cluster_banded(features: np.ndarray, similarity_threshold: float, half: int) -> np.ndarray:
    """Single-linkage clustering, without scipy: connected components of the
    graph with an edge between candidates i < j (only if j - i <= half) when
    their cosine similarity is >= similarity_threshold. Union-find with path
    compression; similarity computed in row-blocks bounded to ~256 MB even
    when half == n - 1 (select_window=0, global mode).
    Returns int64 cluster labels, renumbered 0..K-1 in order of first
    (lowest-index) candidate; length == features.shape[0].
    """
    n = features.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if n > 1 and half > 0:
        block_rows = _pick_block_rows(n, half)
        row = 0
        while row < n:
            r_end = min(n, row + block_rows)
            col_start = row + 1
            col_end = min(n, (r_end - 1) + half + 1)
            if col_start < col_end:
                block = features[row:r_end] @ features[col_start:col_end].T
                for local_i in range(r_end - row):
                    i = row + local_i
                    j_hi = min(n - 1, i + half)
                    if j_hi <= i:
                        continue
                    js = np.arange(i + 1, j_hi + 1)
                    sims = block[local_i, js - col_start]
                    for j in js[sims >= similarity_threshold]:
                        union(i, int(j))
            row = r_end

    labels = np.empty(n, dtype=np.int64)
    next_label = 0
    root_to_label = {}
    for i in range(n):
        root = find(i)
        label = root_to_label.get(root)
        if label is None:
            label = next_label
            root_to_label[root] = label
            next_label += 1
        labels[i] = label
    return labels


def _build_clusters(frame_ids: list, filenames: list, labels: np.ndarray, mean_l: np.ndarray) -> list:
    """Group candidate indices by cluster label and pick one representative
    per cluster: the member whose mean-L is closest to the cluster's median
    mean-L (least over/underexposed candidate); ties broken by the lowest
    frame id. Returns clusters sorted by representative frame id."""
    groups = defaultdict(list)
    for idx, label in enumerate(labels):
        groups[int(label)].append(idx)

    clusters = []
    for indices in groups.values():
        l_vals = [float(mean_l[i]) for i in indices]
        median_l = float(np.median(l_vals))
        rep_idx = min(indices, key=lambda i: (abs(float(mean_l[i]) - median_l), i))
        clusters.append({
            "ref_file": filenames[rep_idx],
            "representative_frame_id": frame_ids[rep_idx],
            "member_frame_ids": sorted(frame_ids[i] for i in indices),
        })

    clusters.sort(key=lambda c: c["representative_frame_id"])
    return clusters


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _mean_l_channel(img: Image.Image) -> float:
    """Mean L value (CIE Lab) of an RGB PIL image, used only to rank cluster
    members (least over/underexposed candidate)."""
    import cv2
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return float(lab[:, :, 0].mean())


def _thumbnail_base64(img: Image.Image, max_px: int = 192, quality: int = 70) -> str:
    """Base64-encoded JPEG thumbnail, for cluster_debug.html only."""
    thumb = img.copy()
    thumb.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    thumb.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _transfer_file(src: str, dst: str, move: bool) -> None:
    """Move (removes src) or copy (preserves src, copy2 keeps metadata) a
    single representative file from ref_framedir to out_framedir."""
    import shutil
    if move:
        shutil.move(src, dst)
    else:
        shutil.copy2(src, dst)


def _write_debug_html(clusters: list, frame_ids: list, thumbs_b64: list, out_framedir: str) -> str:
    """Write cluster_debug.html to out_framedir: one section per multi-member
    cluster, thumbnails of every member, representative highlighted.
    Single-member clusters are omitted here (there is no merge decision to
    review - a singleton has nothing to compare itself against - and on a
    real clip they vastly outnumber the multi-member ones, e.g. 746 of 1058
    clusters on a real 2196-candidate test); they are still kept, unchanged,
    in cluster_map.json and out_framedir. Adapted to the base64 JPEG 
    thumbnails already collected while streaming.
    """
    id_to_thumb = {frame_ids[i]: thumbs_b64[i] for i in range(len(frame_ids))}

    multi_member = [c for c in clusters if len(c["member_frame_ids"]) > 1]
    n_singletons = len(clusters) - len(multi_member)
    ordered = sorted(multi_member, key=lambda c: c["representative_frame_id"])

    sections = []
    for cluster_idx, c in enumerate(ordered):
        rep_id = c["representative_frame_id"]
        member_ids = c["member_frame_ids"]
        thumbs_html = []
        for fid in member_ids:
            b64 = id_to_thumb.get(fid)
            if b64 is None:
                continue
            border = ' style="border:3px solid #e63946;"' if fid == rep_id else ""
            thumbs_html.append(
                f'<figure style="display:inline-block;margin:4px;text-align:center">'
                f'<img src="data:image/jpeg;base64,{b64}"{border} />'
                f'<figcaption style="font-size:11px">#{fid:06d}'
                f'{"&nbsp;&#9733;" if fid == rep_id else ""}</figcaption>'
                f'</figure>'
            )
        sections.append(
            f'<section style="border:1px solid #ccc;margin:16px 0;padding:12px;">'
            f'<h2 style="margin:0 0 8px">Cluster {cluster_idx + 1} '
            f'&mdash; rep: <code>{c["ref_file"]}</code> '
            f'({len(member_ids)} member{"s" if len(member_ids) != 1 else ""})</h2>'
            f'{"".join(thumbs_html)}'
            f'</section>'
        )

    html = (
        "<!DOCTYPE html>\n<html><head>\n"
        '<meta charset="utf-8">'
        "<title>Reference Frame Selection - Cluster Debug View</title>\n"
        "</head><body>\n"
        "<h1>Reference Frame Selection &mdash; Cluster Debug View</h1>\n"
        "<p>Red border = cluster representative (median-L selection). "
        f"Showing {len(ordered)} multi-member cluster(s); {n_singletons} single-member "
        "cluster(s) have nothing to compare and are omitted here (kept unchanged in "
        "cluster_map.json / out_framedir).</p>\n"
        + "\n".join(sections)
        + "\n</body></html>\n"
    )

    debug_path = os.path.join(out_framedir, "cluster_debug.html")
    with open(debug_path, "w", encoding="utf-8") as f:
        f.write(html)
    return debug_path
