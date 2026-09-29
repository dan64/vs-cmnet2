# Imports
import sys
import vapoursynth as vs
core = vs.core

import vscmnet2 as cmnet2

# --- parameters (edit here between runs) ------------------------------------
REF_FRAMEDIR = r"references"
OUT_FRAMEDIR = r"references_filtered"
SIMILARITY_THRESHOLD = 0.95
SELECT_WINDOW = 50
DEVICE = "cuda"
# DRY_RUN=True: only cluster_map.json (and cluster_debug.html) are written,
# no ref_* file is copied to OUT_FRAMEDIR. Set to False once the clustering
# looks right in cluster_debug.html and you want the representatives copied
# for real production data.
DRY_RUN = True
DEBUG_HTML = True
# If True, representative files are MOVED out of REF_FRAMEDIR instead of
# copied (ref_qwen loses those files). Candidates that lost their cluster
# always stay in REF_FRAMEDIR either way. Ignored while DRY_RUN=True.
MOVE_FILES = False

# vs_select_reference_frames() is the VapourSynth-friendly entry point (see
# vscmnet2/cmnet2_refselect.py): it returns a placeholder clip whose length
# equals the number of candidates to process, and does its real work lazily,
# one candidate at a time, only as frames of that clip are requested - same
# style as vs_sc_export_frames(). No result dict here: everything useful is
# already written to cluster_map.json / cluster_debug.html inside
# OUT_FRAMEDIR once the whole clip has been requested (e.g. via vspipe, or by
# playing/exporting it - not just previewing a single frame in vsedit).
clip = cmnet2.vs_select_reference_frames(
    ref_framedir=REF_FRAMEDIR,
    out_framedir=OUT_FRAMEDIR,
    similarity_threshold=SIMILARITY_THRESHOLD,
    select_window=SELECT_WINDOW,
    device=DEVICE,
    dry_run=DRY_RUN,
    debug_html=DEBUG_HTML,
    move_files=MOVE_FILES,
)
clip.set_output()
