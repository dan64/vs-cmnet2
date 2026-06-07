"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-04-08
version: 
LastEditors: Dan64
LastEditTime: 2026-06-07
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
module containing the main functions to colorize the frames with CMNET2.
"""
import vapoursynth as vs
import math
import torch
from functools import partial

from ..colormnet2 import vs_colormnet2_remote, vs_colormnet2_local
from ..colormnet2 import vs_colormnet2dit_remote, vs_colormnet2dit_local
from .vsimage_engine import CMNET2ditEngine

from .constants import *

def vs_colormnet2(clip: vs.VideoNode, clip_ref: vs.VideoNode, clip_sc: vs.VideoNode, image_size: int = -1,
                  enable_resize: bool = False, frame_propagate: bool = True, render_vivid: bool = True,
                  max_memory_frames: int = 0, encode_mode: int = 0, ref_weight: float = 1.0,
                  sc_framedir: str = None, retry_perm_share_threshold: float = 0.25,
                  retry_model: int = 1) -> vs.VideoNode:
    """Colorize a clip using CMNET2 (exemplar-based, sliding permanent-memory).

    :param clip:               B&W source clip (RGB24).
    :param clip_ref:           Coloured reference clip carrying scene-change frame props.
    :param clip_sc:            Auxiliary clip for ref-merge scene detection (may be None).
    :param image_size:         Inference resolution override. -1 = use clip size.
    :param enable_resize:      Enable internal upscaling for higher colour accuracy.
    :param frame_propagate:    If True, propagate colours to non-reference frames.
    :param render_vivid:       If True, increase output saturation by ~15%.
    :param max_memory_frames:  Sliding window size (number of reference frames). 0 = DEF_XRF_WINDOW_SIZE.
    :param encode_mode:        0 = remote (XML-RPC subprocess), 1 = local (in-process).
    :param ref_weight:         Blend weight for reference frames [0, 1].
    :param sc_framedir:        Path to a directory of pre-saved reference images for direct access (ref_mode=0).
    :param retry_perm_share_threshold:  Threshold on perm_share below which reference_frame_missing() returns True.
                                        Default 0.0 (disabled).
    :param retry_model:         Model used by the retry path to colorize missing reference frames.
                                     1 = DiT fp4, 2 = DiT int4. Default 1.
    :return:                   Colourised clip (RGB24).
    """
    # max_memory_frames acts as window_size for the sliding perm_mem; default to DEF_XRF_WINDOW_SIZE
    if max_memory_frames is None or max_memory_frames == 0:
        max_memory_frames = DEF_XRF_WINDOW_SIZE
    max_memory_frames = max(2, math.trunc(max_memory_frames / 2) * 2)

    match encode_mode:
        case 0:
            return vs_colormnet2_remote(clip, clip_ref, clip_sc, image_size, enable_resize, frame_propagate,
                                        render_vivid, max_memory_frames, ref_weight, sc_framedir,
                                        retry_perm_share_threshold=retry_perm_share_threshold,
                                        retry_model=retry_model)
        case 1:
            return vs_colormnet2_local(clip, clip_ref, clip_sc, image_size, enable_resize, frame_propagate,
                                       render_vivid, max_memory_frames, ref_weight, sc_framedir,
                                       retry_perm_share_threshold=retry_perm_share_threshold,
                                       retry_model=retry_model)
        case _:
            raise vs.Error(f"vs_cmnet2: encode_mode must be 0 or 1, got {encode_mode}")

# Default parameters for HAVCditEngine when the caller does not provide them.
# Keys mirror HAVCditEngine.__init__() arguments exactly.
_DEF_DIT_ENGINE_PARAMS: dict = {
    "host"                 : "127.0.0.1",
    "port"                 : 8765,
    "model_name"           : "nunchaku-qwen",
    "model_precision"      : "fp4",
    "model_rank"           : "32",
    "model_inference_steps": "4",
    "cache_dir"            : "",
    "full_model_path"      : "",
    "prompt"               : (
        "Colorize this image, natural colors. "
        "Strictly preserve all shapes, edges and background details."
    ),
    "steps"   : 2,
    "img_size": 0,
}

def vs_colormnet2dit(clip: vs.VideoNode, clip_ref: vs.VideoNode,
                     dit_engine_params: dict = None,  image_size: int = -1,
                     enable_resize: bool = False, frame_propagate: bool = False,
                     render_vivid: bool = False, max_memory_frames: int = 0,
                     encode_mode: int = 0, retry_perm_share_threshold: float = 0.0,
                     retry_model: int = 0) -> vs.VideoNode:
    """Colorize a clip using CMNET2-DIT with a sliding permanent-memory window.

    DIT variant of vs_colormnet2(): reference frames are treated as B&W and
    colorized by CMNET2ditEngine (a DiT-based model via RPC) before being loaded
    into CMNET2 permanent memory.  Colorization always runs in pairs to exploit
    CMNET2ditEngine.colorize_image_pair(); a single colorize_image() call handles
    any odd leftover at the end of the reference list.

    :param clip:                        B&W source clip (RGB24).
    :param clip_ref:                    B&W clip carrying scene-change props and
                                        reference frames.  In HAVC_cmnet2dit this
                                        is derived from the input clip itself.
    :param dit_engine_params:           Dict of keyword arguments forwarded to
                                        HAVCditEngine().  Missing keys fall back to
                                        _DEF_DIT_ENGINE_PARAMS defaults.
                                        Recognised keys (all optional):
                                            host, port, model_name, model_precision,
                                            model_rank, model_inference_steps,
                                            cache_dir, full_model_path,
                                            prompt, steps, img_size.
    :param image_size:                  CMNET2 inference resolution override (-1 = clip size).
    :param enable_resize:               Enable internal CMNET2 upscaling.
    :param frame_propagate:             If True, propagate colours from reference frames.
    :param render_vivid:                If True, apply a gentle hue/saturation boost (~15%).
    :param max_memory_frames:           Sliding window size.  0 → DEF_XRF_WINDOW_SIZE.
                                        Rounded down to nearest even number internally.
    :param encode_mode:                 0 = remote CMNET2 backend (XML-RPC subprocess).
                                        1 = local CMNET2 backend (in-process).
    :param retry_perm_share_threshold:  CMNET2 retry threshold (0.0 = disabled, default).
    :param retry_model:                 If retry_perm_share_threshold > 0, model used to colorize missing (default: 1)
                                        reference frames. Allowed values are:
                                             0 = HAVC (DeOldify + DDColor),
                                             1 = DiT fp4,
                                             2 = DiT int4.
    :return:                            Colourised clip (RGB24).
    """
    # Resolve window size: default → DEF_XRF_WINDOW_SIZE, then force even.
    if max_memory_frames is None or max_memory_frames == 0:
        max_memory_frames = DEF_XRF_WINDOW_SIZE
    max_memory_frames = max(2, math.trunc(max_memory_frames / 2) * 2)

    engine_params = dict(_DEF_DIT_ENGINE_PARAMS)
    if dit_engine_params:
        engine_params.update(dit_engine_params)

    dit_engine = CMNET2ditEngine(**engine_params)

    match encode_mode:
        case 0:
            return vs_colormnet2dit_remote(
                clip, clip_ref, dit_engine,
                image_size, enable_resize, frame_propagate,
                render_vivid, max_memory_frames,
                retry_perm_share_threshold=retry_perm_share_threshold,
                retry_model=retry_model)
        case 1:
            return vs_colormnet2dit_local(
                clip, clip_ref, dit_engine,
                image_size, enable_resize, frame_propagate,
                render_vivid, max_memory_frames,
                retry_perm_share_threshold=retry_perm_share_threshold,
                retry_model=retry_model)
        case _:
            raise vs.Error(f"vs_cmnet2dit: encode_mode must be 0 or 1, got {encode_mode}")
