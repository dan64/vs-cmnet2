"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2026-06-07
version: 
LastEditors: Dan64
LastEditTime: 2026-06-07
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
main Vapoursynth wrapper to pytorch-based coloring filter CMNET2.
CMNET2: https://github.com/dan64/cmnet2
"""
from __future__ import annotations

import os

from .colormnet2 import vs_colormnet2_remote
from .vsslib.constants import DEF_XRF_WINDOW_SIZE

os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["NUMEXPR_MAX_THREADS"] = "8"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TORCH_LOGS"] = "-all"

import pathlib
import math


from .cmnet2_utils import convert_format_RGB24, restore_format, VIDEO_EXTENSIONS, get_ref_number
from .vsslib.mcomb import vs_combine_models, vs_ext_reference_clip
from .vsslib.vsfilters import vs_simple_merge, vs_sc_colormap, vs_sc_dark_tweak
from .vsslib.vsfilters import  vs_sc_chroma_bright_tweak, vs_recover_clip_luma
from .vsslib.vsfilters import vs_get_clip_frame
from .vsslib.vsmodels import vs_colormnet2, vs_colormnet2dit
from .vsslib.vsplugins import load_LSMASHSource_plugin
from .vsslib.vsutils import vs_sc_export_frames, vs_list_export_frames, CMNET2_LogMessage, MessageType
from .vsslib.vsresize import get_render_size
from .vsslib.vsscdect import SceneDetectFromDir, SceneDetect, CopySCDetect
from .vsslib.vsscdect import get_sc_props
from .vsslib.vsscdetect_edge import SceneDetectEdges
from .colormnet2 import vs_colormnet2_range

from .vsslib import constants as constants

__version__ = "1.0.2"

import warnings
import logging

from typing import Sequence

warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
warnings.filterwarnings("ignore", category=UserWarning,
                        message="The parameter 'pretrained' is deprecated since 0.13 and may be removed in the "
                                "future, please use 'weights' instead.")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum or `None`.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="torch.nn.utils.weight_norm is deprecated.*?")
warnings.filterwarnings("ignore", category=UserWarning, message="Conversion from CIE-LAB,*?")
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Torch was not compiled with flash attention.*?")

warnings.filterwarnings("ignore", category=FutureWarning, message=".torch.cuda.amp.custom_fwd.*?")
warnings.filterwarnings("ignore", category=FutureWarning, message="Arguments other than a weight enum or `None`.*?")
warnings.filterwarnings("ignore", category=FutureWarning, message=".You are using `torch.load`.*?")

warnings.simplefilter(action='ignore', category=FutureWarning)

package_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(package_dir, "models")

# configuring torch
import torch
torch.backends.cudnn.benchmark = True

import vapoursynth as vs

def vs_cmnet2(clip: vs.VideoNode = None, clip_ref: vs.VideoNode = None, method: int = 4, render_speed: str = 'auto',
              render_vivid: bool = False, sc_framedir: str = None, dark: bool = False, dark_p: list = (0.2, 0.8),
              smooth: bool = False, smooth_p: list = (0.3, 0.7, 0.9, 0.0, "none"), colormap: str = "none",
              encode_mode: int = 0, max_memory_frames: int = 0, ref_mode: int = 1, retry_threshold: float = 0.0,
              retry_model: int = 1, torch_dir: str = model_dir) -> vs.VideoNode:
    """CMNET2 colorization filter
    :param clip:                Clip to process, any clip format is supported
    :param clip_ref:            Clip containing the reference frames (necessary if method=0,1,2,5,6)
    :param method:              Method to use to generate reference frames (RF).
                                        3 = external RF same as video
                                        4 = external RF different from video
                                        5 = external ClipRef same as video
                                        6 = external ClipRef different from video
    :param render_speed:        Preset to control the render method and speed:
                                Allowed values are:
                                        'Auto'   : will be automatically assigned the optimal render size (default)
                                        'Fast'   : colors are more washed out
                                        'Medium' : colors are a little washed out
                                        'Slow'   : colors are a little more vivid
                                        'Slower' : colors are more accurate (usually is very slow)
    :param render_vivid:        If True, the saturation will be increased by about 15%. Default: False
    :param sc_framedir:         If set, define the directory where are stored the reference frames. If only_ref_frames=True,
                                and method=0 this directory will be written with the reference frames used by the filter.
                                if method!=0 the directory will be read to create the reference frames that will be used
                                by "Exemplar-based" Video Colorization. The reference frame name must be in the
                                format: ref_nnnnnn.[jpg|png], for example the reference frame 897 must be
                                named: ref_000897.png. With methods 5,6 this parameters can be the path to a video clip.
                                NOTE: When used with method in (1, 2, 3, 4), reference frames are read directly
                                      from this directory instead of being re-evaluated through the VapourSynth
                                      pipeline for each preload/slide operation. This is significantly faster
                                      but means that reference-frame filters (colormap, dark, smooth) are NOT
                                      applied to the permanent-memory refs — they are applied only to the VS
                                      clip_ref used for the runtime merge. If you need those filters applied
                                      uniformly, apply equivalent post-processing to the final colored clip.
    :param dark:                Enable/disable darkness filter (only on ref-frames), range [True,False]
    :param dark_p:              Parameters for darken the clip's dark portions, which sometimes are wrongly colored by the color models
                                      [0] : dark_threshold, luma threshold to select the dark area, range [0.1-0.5] (0.01=1%)
                                      [1] : dark_amount: amount of desaturation to apply to the dark area, range [0-1]
                                      [2] : "chroma range" parameter (optional), if="none" is disabled (see the README)
    :param smooth:              Enable/disable chroma smoothing (only on ref-frames), range [True, False]
    :param smooth_p:            parameters to adjust the saturation and "vibrancy" of the clip.
                                      [0] : dark_threshold, luma threshold to select the dark area, range [0-1] (0.01=1%)
                                      [1] : white_threshold, if > dark_threshold will be applied a gradient till white_threshold, range [0-1] (0.01=1%)
                                      [2] : dark_sat, amount of de-saturation to apply to the dark area, range [0-1]
                                      [3] : dark_bright, darkness parameter it used to reduce the "V" component in "HSV" colorspace, range [0, 1]
                                      [4] : "chroma range" parameter (optional), if="none" is disabled (see the README)
    :param colormap:            Direct hue/color mapping (only on ref-frames), without luma filtering, using the "chroma adjustment"
                                parameter, if="none" is disabled.
    :param encode_mode:         Parameter used by ColorMNet to define the encode mode strategy.
                                Available values are:
                                     0: remote encoding. The frames will be colored by a thread outside Vapoursynth.
                                                         This option has no GPU memory limitation and fully exploits
                                                         the long-term frame memory. It is the faster encode method
                                                         (default). All available reference frames are used via the
                                                         sliding permanent-memory window.
                                     1: local encoding.  The frames will be colored inside the Vapoursynth environment.
                                                         Useful when remote encoding is not available or when a single
                                                         process is preferred.
    :param max_memory_frames:   Window size for the sliding permanent-memory of ColorMNet2.
                                Defines how many reference frames are held in the model's permanent memory at any
                                given time. Must be an even number and must not exceed the number of available
                                reference frames. The window slides forward automatically as colorization advances.
                                Suggested values: min=10, max=500.
                                If = 0 (default) will be set to DEF_XRF_WINDOW_SIZE (20).
    :param ref_mode:            Mode selected to access to the external reference frames.
                                Allowed values are:
                                    0: will use direct access to reference frame folder
                                    1: will use Vapoursynth clips to access to reference frames (default)
    :param retry_threshold:     Threshold used to identify frames that may benefit from an additional
                                reference frame. Range [0.0, 1.0], default 0.0 (disabled).
                                High values (> 0.3) trigger more retry, while lower values (< 0.3) trigger less retry.
                                Suggested value in the range: 0.20-0.35.
    :param retry_model:         If retry_threshold > 0 it represents the model used to colorize the missing
                                reference frames. Allowed values, are:
                                     1 : Model DiT colorization with model_precision = "fp4" (RTX 50-Series)
                                     2 : Model DiT colorization with model_precision = "int4" (RTX 30/40-Series)
                                For the models 1 and 2 is necessary to run the DiT Server as explained in the docstring
                                of vs_cmnet2dit(). In the case the DiT Server is not running will be used the model 0
                                (Model CMNET2). Range [0, 1, 2], default = 0
    :param torch_dir:           torch hub dir location, default is model directory, if set to None will switch
                                to torch cache dir
    """
    # disable packages warnings
    disable_warnings()
    # static variables
    only_ref_frames: bool = False
    if not torch.cuda.is_available():
        CMNET2_LogMessage(MessageType.EXCEPTION, "vs_cmnet2: CUDA is not available")

    clip, orig_fmt = convert_format_RGB24(clip)
    if clip_ref is not None:
        clip_ref, orig_fmt_r = convert_format_RGB24(clip_ref)

    if method not in range(7):
        CMNET2_LogMessage(MessageType.EXCEPTION, "vs_cmnet2: method must be in range [3-6]")

    if method in (0, 1, 2):
        CMNET2_LogMessage(MessageType.EXCEPTION, "vs_cmnet2: method must be in range [3-6]")

    if torch_dir is not None:
        torch.hub.set_dir(torch_dir)

    # static params
    enable_resize = False
    # unpack dark
    dark_enabled = dark
    dark_threshold = dark_p[0]
    dark_amount = dark_p[1]
    if len(dark_p) > 2:
        dark_hue_adjust = dark_p[2]
    else:
        dark_hue_adjust = 'none'

    # unpack chroma_smoothing
    chroma_smoothing_enabled = smooth
    black_threshold = smooth_p[0]
    white_threshold = smooth_p[1]
    dark_sat = smooth_p[2]
    dark_bright = -smooth_p[3]  # change the sign to reduce the bright
    if len(smooth_p) > 4:
        chroma_adjust = smooth_p[4]
    else:
        chroma_adjust = 'none'

    # define colormap
    colormap = colormap.lower()
    colormap_enabled = (colormap != "none" and colormap != "")
    ref_weight = 1.0
    clip_sc = None
    if method != 0 and not (sc_framedir is None):
        ref_frame_ext = method in (2, 4)
        merge_ref_frame = method in (1, 2)
        clip = SceneDetectFromDir(clip, sc_framedir=sc_framedir, merge_ref_frame=merge_ref_frame,
                                  ref_frame_ext=ref_frame_ext)
    else:
        clip = CopySCDetect(clip, clip_ref)

    clip_orig = clip
    # when reference frames exist on disk, read them directly (bypasses VS pipeline
    # re-evaluation per reference — significant speedup on preload/slide).
    # The clip_ref VS path is still used for the merge at runtime (1 get_frame per
    # output frame, not per reference).
    use_dir_refs = (ref_mode == 0
                    and method in (1, 2, 3, 4)
                    and sc_framedir is not None
                    and os.path.isdir(sc_framedir))

    # if user explicitly requested ref_mode=0 but the conditions aren't met,
    # warn and fall back to VS mode silently — alternative: raise exception
    if ref_mode == 0 and not use_dir_refs:
        CMNET2_LogMessage(MessageType.WARNING,
                        "vs_cmnet2: ref_mode=0 (direct) requested but not applicable "
                        "(requires method in (1,2,3,4) and valid sc_framedir). "
                        "Falling back to VS clip mode.")

    if method != 0 and not (sc_framedir is None):
        clip_ref = vs_ext_reference_clip(clip, sc_framedir=sc_framedir)

    d_size = get_render_size(clip.width, clip.height, render_speed=render_speed.lower())
    clip = clip.resize.Spline36(width=d_size[0], height=d_size[1])
    clip_ref = clip_ref.resize.Spline36(width=d_size[0], height=d_size[1])
    # when reference frames are loaded directly from disk, filters on clip_ref
    # (colormap, dark, smooth) are redundant: perm_mem refs bypass them entirely,
    # and applying them only to the runtime merge clip creates an inconsistency
    # that degrades merge quality. Skip them.
    if use_dir_refs:
        if colormap_enabled or dark_enabled or chroma_smoothing_enabled:
            CMNET2_LogMessage(MessageType.WARNING,
                            "vs_cmnet2: ref-frame filters (colormap/dark/smooth) "
                            "are ignored in ref_mode=0. Apply equivalent post-"
                            "processing to the colored clip if needed.")
        colormap_enabled = False
        dark_enabled = False
        chroma_smoothing_enabled = False

    if colormap_enabled:
        clip_ref = vs_sc_colormap(clip_ref, colormap=colormap)

    if dark_enabled:
        clip_ref = vs_sc_dark_tweak(clip_ref, dark_threshold=dark_threshold, dark_amount=dark_amount,
                                    dark_hue_adjust=dark_hue_adjust.lower())

    if chroma_smoothing_enabled:
        clip_ref = vs_sc_chroma_bright_tweak(clip_ref, black_threshold=black_threshold, white_threshold=white_threshold,
                                             dark_sat=dark_sat, dark_bright=dark_bright,
                                             chroma_adjust=chroma_adjust.lower())
    ref_same_as_video = False
    clip_colored = vs_colormnet2(clip, clip_ref, clip_sc, image_size=-1, enable_resize=enable_resize,
                                            encode_mode=encode_mode, max_memory_frames=max_memory_frames,
                                            frame_propagate=ref_same_as_video, render_vivid=render_vivid,
                                            ref_weight=ref_weight, sc_framedir=sc_framedir if use_dir_refs else None,
                                            retry_perm_share_threshold=retry_threshold, retry_model=retry_model)

    clip_resized = clip_colored.resize.Spline36(width=clip_orig.width, height=clip_orig.height)
    # restore original resolution details, 5% faster than ShufflePlanes()
    clip_new = vs_recover_clip_luma(clip_orig, clip_resized)
    return restore_format(clip_new, orig_fmt)


def vs_cmnet2_recolor(clip: vs.VideoNode = None, method: int = 4, render_speed: str = 'auto',
                      render_vivid: bool = False, ref_framedir: str = None, ref_start_path: str = None,
                      ref_end_path: str = None, max_memory_frames: int = 0, retry_threshold: float = 0.0,
                      retry_model: int = 1, torch_dir: str = model_dir) -> vs.VideoNode:
    """CMNET2 colorization filter with colorization limited by a range of reference frames
    :param clip:                Colorized clip to process, any clip format is supported
    :param method:              Method to use to generate reference frames (RF).
                                        3 = external RF same as video
                                        4 = external RF different from video
    :param render_speed:        Preset to control the render method and speed:
                                Allowed values are:
                                        'Auto'   : will be automatically assigned the optimal render size (default)
                                        'Fast'   : colors are more washed out
                                        'Medium' : colors are a little washed out
                                        'Slow'   : colors are a little more vivid
                                        'Slower' : colors are more accurate (usually is very slow)
    :param render_vivid:        If True, the saturation will be increased by about 15%. Default: False
    :param ref_framedir:         Define the directory where are stored the colorized reference frames.
                                The reference frame name must be in the format: ref_nnnnnn.[jpg|png],
                                for example the reference frame 897 must be named: ref_000897.png.
                                NOTE: The reference frames are read directly from this directory.
    :param max_memory_frames:   Window size for the sliding permanent-memory of ColorMNet2.
                                Defines how many reference frames are held in the model's permanent memory at any
                                given time. Must be an even number and must not exceed the number of available
                                reference frames. The window slides forward automatically as colorization advances.
                                Suggested values: min=10, max=500.
                                If = 0 (default) will be set to DEF_XRF_WINDOW_SIZE (20).
    :param ref_start_path:      Path to the first frame to be used for the clip re-colorization. Must be in the format
                                ref_nnnnnn.[jpg|png] and need to be stored in the same path defined in ref_framedir.
    :param ref_end_path:        Path to the last frame to be used for the clip re-colorization. Must be in the format
                                ref_nnnnnn.[jpg|png] and need to be stored in the same path defined in ref_framedir.
    :param retry_threshold:     Threshold used to identify frames that may benefit from an additional
                                reference frame. Range [0.0, 1.0], default 0.0 (disabled).
                                High values (> 0.3) trigger more retry, while lower values (< 0.3) trigger less retry.
                                Suggested value in the range: 0.20-0.35.
    :param retry_model:         If retry_threshold > 0 it represents the model used to colorize the missing
                                reference frames. Allowed values, are:
                                     1 : Model DiT colorization with model_precision = "fp4" (RTX 50-Series)
                                     2 : Model DiT colorization with model_precision = "int4" (RTX 30/40-Series)
                                For the models 1 and 2 is necessary to run the DiT Server as explained in the docstring
                                of vs_cmnet2dit(). In the case the DiT Server is not running will be used the model 0
                                (Model CMNET2). Range [0, 1, 2], default = 0
    :param torch_dir:           torch hub dir location, default is model directory, if set to None will switch
                                to torch cache dir
    """
    # disable packages warnings
    disable_warnings()
    if not torch.cuda.is_available():
        CMNET2_LogMessage(MessageType.EXCEPTION, "vs_cmnet2_recolor: CUDA is not available")

    if method not in (3, 4):
        CMNET2_LogMessage(MessageType.EXCEPTION, "vs_cmnet2_recolor: method must be in range [3-4]")

    if ref_framedir is None:
        CMNET2_LogMessage(MessageType.EXCEPTION, "vs_cmnet2_recolor: 'ref_framedir' must be provided")

    ref_start = get_ref_number(ref_start_path)
    if ref_start is None:
        CMNET2_LogMessage(MessageType.EXCEPTION, f"vs_cmnet2_recolor: failed to get ref number form {ref_start_path}")

    ref_end = get_ref_number(ref_end_path)
    if ref_end is None:
        CMNET2_LogMessage(MessageType.EXCEPTION, f"vs_cmnet2_recolor: failed to get ref number form {ref_end_path}")

    if ref_end <= ref_start:
        CMNET2_LogMessage(MessageType.EXCEPTION, f"vs_cmnet2_recolor: {ref_end_path} must be greater than {ref_start_path}")

    if torch_dir is not None:
        torch.hub.set_dir(torch_dir)

    # Split: only the middle segment goes through CMNET2
    # ref_start and ref_end are frame numbers (inclusive on both sides)
    clip_mid = clip[ref_start:ref_end + 1]
    has_head = ref_start > 0
    has_tail = ref_end + 1 < clip.num_frames

    clip_mid, orig_fmt = convert_format_RGB24(clip_mid)
    clip_mid_orig = clip_mid
    orig_w, orig_h = clip_mid.width, clip_mid.height
    d_size = get_render_size(orig_w, orig_h, render_speed=render_speed.lower())
    clip_mid = clip_mid.resize.Spline36(width=d_size[0], height=d_size[1])

    if max_memory_frames is None or max_memory_frames == 0:
        max_memory_frames = DEF_XRF_WINDOW_SIZE
    max_memory_frames = max(2, math.trunc(max_memory_frames / 2) * 2)

    ref_same_as_video = (method == 3)
    clip_mid_colored = vs_colormnet2_range(clip_mid, clip_ref=None, frame_propagate=ref_same_as_video,
                                 render_vivid=render_vivid, max_memory_frames=max_memory_frames,
                                 sc_framedir=ref_framedir, ref_range=(ref_start, ref_end),
                                 retry_perm_share_threshold=retry_threshold, retry_model=retry_model,
                                 frame_offset=ref_start)

    clip_mid_colored = clip_mid_colored.resize.Spline36(width=orig_w, height=orig_h)
    clip_mid_new = vs_recover_clip_luma(clip_mid_orig, clip_mid_colored)
    clip_mid_new = restore_format(clip_mid_new, orig_fmt)

    # Splice: head untouched + recolored mid + tail untouched
    if has_head:
        if has_tail:
            return clip[:ref_start] + clip_mid_new + clip[ref_end + 1:]
        else:
            return clip[:ref_start] + clip_mid_new
    else:
        if has_tail:
            return clip_mid_new + clip[ref_end + 1:]
        else:
            return clip_mid_new


def vs_cmnet2dit(clip: vs.VideoNode = None,
                   render_speed: str = 'auto',
                   render_vivid: bool = False,
                   sc_thresh: float = 0.035,
                   sc_tht_ssim: float= 0.80,
                   sc_min_int: int = 25,
                   sc_tht_offset: int = 2,
                   sc_min_freq: int = 0,
                   max_memory_frames: int = 20,
                   dit_engine_params: dict = None,
                   retry_threshold: float = 0.0,
                   retry_model: int = 1,
                   torch_dir: str = model_dir) -> vs.VideoNode:
    """CMNET2-DIT colorization filter.
    Like HAVC_cmnet2() but designed for B&W reference frames: scene-change
    frames extracted from the input clip are colorized by a DiT-based model
    (DiT Engine, accessed via RPC) *before* being loaded into CMNET2
    permanent memory.  This makes HAVC_cmnet2dit() self-contained, no
    eparate pre-colored reference clip is needed.
    The DiT colorization of reference frames always runs in pairs
    (colorize_image_pair()) to exploit the DiT model's batched forward pass.
    A single colorize_image() call handles any odd leftover reference frame at
    the end of the clip.
    :param clip:                B&W source clip; any format is accepted and
                                converted internally to RGB24.
    :param render_speed:        Preset controlling CMNET2 render resolution.
                                Allowed values (case-insensitive):
                                    'Auto'   – optimal size chosen automatically (default)
                                    'Fast'   – more washed-out colours
                                    'Medium' – slightly washed out
                                    'Slow'   – slightly more vivid
                                    'Slower' – most accurate (usually very slow)
    :param render_vivid:        If True, apply a ~15% saturation boost after
                                colorization. Default: False.
    :param sc_thresh:           Scene edges-detection threshold used to select
                                reference frames from the input clip.
                                Range [0.01, 0.15]. Default: 0.035.
    :param sc_tht_ssim:         Threshold used by the SSIM (Structural Similarity Index Metric) selection filter.
                                If > 0, will be activated a filter that will improve the scene-change detection,
                                by discarding images that are similar. Suggested values are between 0.35 and 0.85,
                                range [0-1], default 0.80
    :param sc_min_int:          Minimum frame distance between scene changes. Default 25.
    :param sc_tht_offset:       Offset index used for the Scene change detection. The comparison will be performed,
                                between frame[n] and frame[n-offset]. An offset > 1 is useful to detect blended scene
                                change, range[1, 25]. Default = 2.
    :param sc_min_freq:         If > 0 will be generated at least 1 reference frame every "sc_min_freq" frames.
                                range [0-1000], default: 0.
    :param max_memory_frames:   Sliding permanent-memory window size for CMNET2.
                                Automatically rounded down to the nearest even number
                                (required for pair-wise DIT colorization).
                                0 → DEF_XRF_WINDOW_SIZE (20). Suggested: 10–500. Default = 20
    :param dit_engine_params:   Optional dict of keyword arguments forwarded to
                                DiT Engine Server.  Any key not provided falls back
                                to the DiT Engine Server default.  Recognized keys:
                                    host            : RPC server address (default "127.0.0.1")
                                    port            : RPC server port   (default 8765)
                                    model_name      : Nunchaku model name
                                    model_precision : "fp4" (RTX 50xx) or "int4" (RTX 30/40xx)
                                    model_rank      : SVD rank "32" or "128"
                                    model_inference_steps : steps used to pick the model file
                                    cache_dir       : HuggingFace cache directory
                                    full_model_path : absolute path to a local .safetensors
                                    prompt          : text prompt guiding colorization
                                    steps           : inference steps per image (default 2)
                                    img_size        : max long-side in pixels before inference (0 = original size)
    :param retry_threshold:     Threshold used to identify frames that may benefit from
                                 an additional reference frame. Range [0.0, 1.0],
                                 default 0.0 (disabled). Suggested: 0.20-0.35.
    :param retry_model:         If retry_threshold > 0, model used to colorize missing (default: 1)
                                reference frames. Allowed values are:
                                     0 = CMNET2 (DeOldify + DDColor),
                                     1 = DiT fp4,
                                     2 = DiT int4.
    :param torch_dir:           Torch hub directory for CMNET2 model weights.
                                Default: package model directory.
                                Pass None to use the Torch cache directory.
    :return:                    Colorized clip in the same format as the input.
    """
    disable_warnings()
    if not torch.cuda.is_available():
        CMNET2_LogMessage(MessageType.EXCEPTION, "vs_cmnet2dit: CUDA is not available")

    clip, orig_fmt = convert_format_RGB24(clip)
    if torch_dir is not None:
        torch.hub.set_dir(torch_dir)

    # -----------------------------------------------------------------------
    # Scene-change detection — produces ref frames from the B&W input clip
    # -----------------------------------------------------------------------
    # Apply defaults when caller did not provide explicit thresholds.
    if sc_thresh is None:
        ref_thresh = 0.035
    if sc_min_int is None:
        ref_freq = 25

    # forced encode_mode=0 due to memory limitation
    encode_mode = 0
    # Run scene detection on the (resized-later, but prop-only here) clip to
    # obtain _SceneChangePrev props.  clip_ref == clip internally: the B&W
    # input clip is both the content to colorize and the source of reference
    # frames that DiT Engine Server will colorize.
    clip_ref = SceneDetectEdges(clip, threshold=sc_thresh, frequency=sc_min_freq, ssim_threshold=sc_tht_ssim,
                                sc_diff_offset=sc_tht_offset, sc_min_int=sc_min_int, sc_mult_tht=15,
                                tht_white=0.70, tht_black=0.10)
    # Copy scene-change props to the working clip so that downstream VS filters
    # (e.g. vs_recover_clip_luma) can access them if needed.
    clip = CopySCDetect(clip, clip_ref)
    clip_orig = clip
    # No ref-merge in the DIT path.
    ref_same_as_video = False
    clip_sc = None
    # -----------------------------------------------------------------------
    # Resize to model inference resolution
    # -----------------------------------------------------------------------
    enable_resize = False   # static: consistent with HAVC_cmnet2 default
    d_size = get_render_size(clip.width, clip.height, render_speed=render_speed.lower())
    clip = clip.resize.Spline36(width=d_size[0], height=d_size[1])
    clip_ref = clip_ref.resize.Spline36(width=d_size[0], height=d_size[1])
    # -----------------------------------------------------------------------
    # CMNET2-DIT colorization
    # -----------------------------------------------------------------------
    clip_colored = vs_colormnet2dit(
        clip, clip_ref,
        dit_engine_params=dit_engine_params,
        image_size=-1,
        enable_resize=enable_resize,
        encode_mode=encode_mode,
        max_memory_frames=max_memory_frames,
        frame_propagate=ref_same_as_video,
        render_vivid=render_vivid,
        retry_perm_share_threshold=retry_threshold,
        retry_model=retry_model,
    )
    # -----------------------------------------------------------------------
    # Restore original resolution and format
    # -----------------------------------------------------------------------
    clip_resized = clip_colored.resize.Spline36(width=clip_orig.width, height=clip_orig.height)
    # Graft the original luma back onto the coloured chroma for sharpness
    # (~5% faster than ShufflePlanes).
    clip_new = vs_recover_clip_luma(clip_orig, clip_resized)
    return restore_format(clip_new, orig_fmt)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
function with CMNET2 merge methods  
"""


def vs_merge(clipa: vs.VideoNode = None, clipb: vs.VideoNode = None, clip_luma: vs.VideoNode = None, weight: float = 0.5,
               method: int = 2, cmc_p: list = constants.DEF_CMC_p, lmm_p: list = constants.DEF_LMM_p, alm_p: list = constants.DEF_ALM_p,
               crt_p: list = constants.DEF_CRT_p) -> vs.VideoNode:
    """Utility function with the implementation of CMNET2 merge methods
    :param clipa:               first clip to merge, any format is supported
    :param clipb:               second clip to merge, any format is supported
    :param clip_luma:           if specified, clip_luma will be used as source of luma component for the merge. It is an
                                optional parameter, and it is suggested to provide the clip with the best luma
                                resolution between clipa and clipb. It is used only with the methods: 3, 4, 5 and can
                                speed up the filter when it uses these methods.
    :param method:              method used to combine clipa with clipb (default = 2):
                                    0 : clipa only (no merge), if clip_luma is not None is luma merged with clip_luma
                                    1 : clipb only (no merge), if clip_luma is not None is luma merged with clip_luma
                                    2 : Simple Merge (default):
                                        the frames are combined using a weighted merge, where the parameter "weight"
                                        represent the weight assigned to the colors provided by the clipb frames.
                                        If weight = 0 will be returned clipa, if = 1 will be returned clipb.
                                    3 : Constrained Chroma Merge:
                                        The frames are combined by assigning a limit to the amount of difference in
                                        chroma values between clipa and clipb this limit is defined by the threshold
                                        parameter "cmc_tresh".
                                        The limit is applied to the image converted to "YUV". For example when
                                        cmc_tresh=0.2, the chroma values "U","V" of clipb frame will be constrained
                                        to have an absolute percentage difference respect to "U","V" provided by clipa
                                        not higher than 20%. The final limited frame will be merged again with the clipa
                                        frame. With this method is suggested a starting weight > 50% (ex. = 60%).
                                    4 : Luma Masked Merge:
                                        the frames are combined using a masked merge, the pixels of clipb with
                                        luma < "luma_mask_limit" will be filled with the pixels of clipa.
                                        If "luma_white_limit" > "luma_mask_limit" the mask will apply a gradient till
                                        "luma_white_limit". If the parameter "weight" > 0 the final masked frame will
                                        be merged again with the clipa frame.
                                    5 : Adaptive Luma Merge:
                                        The frames are combined by decreasing the weight assigned to clipb when the
                                        luma is below a given threshold given by: luma_threshold. The weight is
                                        calculated using the formula:
                                            merge_weight = max(weight * (luma/luma_threshold)^alpha, min_weight).
                                        For example with: luma_threshold = 0.6 and alpha = 1, the weight assigned to
                                        clipb will start to decrease linearly when the luma < 60% till "min_weight".
                                        For alpha=2, begins to decrease quadratically (because luma/luma_threshold < 1).
                                    6 : Chroma Retention Merge:
                                        Given that the colors provided by deoldify() are more conservative and stable
                                        than the colors obtained with ddcolor(). This function try to restore the
                                        colors of gray pixels provide by deoldify() by using the colors provided
                                        by ddcolor(). The gray pixels are identified by the parameter "tht". Once are
                                        identified the gray pixels are substituted with the desaturated colors in deoldify(),
                                        the level of desaturation is identified by the parameter "sat". It is performed
                                        a "gradient" substitution, i.e. the gray pixels are gradually substituted depending
                                        on the level of gray gradient. The steepness of gradient curve is controlled by
                                        the parameter "alpha". Optionally is possible to resize the frame before the filter
                                        application to speed up the filter by setting True the parameter chroma_resize.
                                    7 : ChromaBound Adaptive
                                        Adaptive version of Constrained-Chroma. In this version the chroma tolerance is
                                        adaptive, i.e., it is applied an approach that will allow more color variation
                                        in textured/complex regions and less in smooth areas. The texture strength is
                                        computed via Laplacian and chroma tolerance is controlled by the following
                                        parameters:
                                              [2] base_tol: int = 20,  # Base chroma tolerance (smooth areas)
                                              [3] max_extra: int = 24,  # Extra tolerance for textured areas
                                    The methods 3, 4 and 7 are similar to Simple Merge, but before the merge with clipa
                                    the clipb frame is limited in the chroma changes (method 3, 7) or limited based
                                    on theluma (method 4). The method 5 is a Simple Merge where the weight decrease
                                    with luma.
    :param weight:              weight given to clipb in all merge methods. If weight = 0 will be returned
                                clipa, if = 1 will be returned clipb. range [0-1] (0.01=1%)
    :param cmc_p:               parameters list for method: "Constrained Chroma Merge", "ChromaBound Adaptive"
                                (see methods 3, 7 for a full explanation):
                                      [0] chroma_threshold (%), range [0-1] (0.01=1%), default = 0.15
                                      [1] red_fix (default = True),  # if true red-regions in dark areas are desaturated
                                      [2] base_tol (default = 20),  # Base chroma tolerance (smooth areas)
                                      [3] max_extra: (default = 24),  # Extra tolerance for textured areas
    :param lmm_p:               parameters for method: "Luma Masked Merge" (see method=4 for a full explanation)
                                   [0] : luma_mask_limit: luma limit for build the mask used in Luma Masked Merge,
                                         range [0-1] (0.01=1%)
                                   [1] : luma_white_limit: the mask will apply a gradient till luma_white_limit,
                                         range [0-1] (0.01=1%)
                                   [2] : luma_mask_sat: if < 1 the clipb dark pixels will substitute with the
                                         desaturated clipa pixels, range [0-1] (0.01=1%)
    :param alm_p:               parameters for method: "Adaptive Luma Merge" (see method=5 for a full explanation)
                                   [0] : luma_threshold: threshold for the gradient merge, range [0-1] (0.01=1%)
                                   [1] : alpha: exponent parameter used for the weight calculation, range [>0]
                                   [2] : min_weight: min merge weight, range [0-1] (0.01=1%)
    :param crt_p:               parameters for method: "Chroma Retention Merge" (see method=6 for a full explanation)
                                   [0] : sat: this parameter allows to change the saturation of colored clip (default = 0.8)
                                   [1] : tht: threshold to identify gray pixels, range[0, 255] (default = 30)
                                   [2] : alpha: parameter used to control the steepness of gradient curve, range [>0] (default = 2.0)
                                   [3] : chroma_resize: if True, the frames will be resized to improve the filter speed (default = False)
    """
    # disable packages warnings
    disable_warnings()
    if clipa is not None and not isinstance(clipa, vs.VideoNode):
        CMNET2_LogMessage(MessageType.EXCEPTION, "CMNET2_merge: this is not a clip: clipa")

    if clipb is not None and not isinstance(clipb, vs.VideoNode):
        CMNET2_LogMessage(MessageType.EXCEPTION, "CMNET2_merge: this is not a clip: clipb")

    if clip_luma is not None and not isinstance(clip_luma, vs.VideoNode):
        CMNET2_LogMessage(MessageType.EXCEPTION, "CMNET2_merge: this is not a clip: clip_luma")

    if method == 0 or weight == 0:
        if clip_luma is not None:
            clipa_sc = clipa
            clipa = _clip_chroma_resize(clip_luma, clipa)
            clipa = CopySCDetect(clipa, clipa_sc)
        return clipa

    if method == 1 or weight == 1:
        if clip_luma is not None:
            clipb_sc = clipb
            clipb = _clip_chroma_resize(clip_luma, clipb)
            clipb = CopySCDetect(clipb, clipb_sc)
        return clipb

    merge_weight = weight
    clip_a, orig_fmt_a = convert_format_RGB24(clipa)
    clip_b, orig_fmt_b = convert_format_RGB24(clipb)
    if method == 2:
        clip_merged = vs_simple_merge(clip_a, clip_b, merge_weight)
        return restore_format(clip_merged, orig_fmt_a)

    if clip_luma is not None:
        rf = min(max(math.trunc(0.4 * clip_luma.width / 16), 16), 32)
        frame_size = min(rf * 16, clip_luma.width)
        clip_a = clip_a.resize.Spline64(width=frame_size, height=frame_size)
        clip_b = clip_b.resize.Spline64(width=frame_size, height=frame_size)

    clip_merged = vs_combine_models(clip_a=clip_a, clip_b=clip_b, method=method, sat=[1, 1],
                                    hue=[0, 0], clipb_weight=merge_weight, CMC_p=cmc_p,
                                    LMM_p=lmm_p, ALM_p=alm_p, CRT_p=crt_p, invert_clips=False)

    if clip_luma is not None:
        clipm_sc = clip_merged
        clip_merged = _clip_chroma_resize(clip_luma, clip_merged)
        clip_merged = CopySCDetect(clip_merged, clipm_sc)

    return restore_format(clip_merged, orig_fmt_a)


def vs_SceneDetect(clip: vs.VideoNode, sc_threshold: float = constants.DEF_THRESHOLD, sc_tht_offset: int = 1,
                     sc_tht_ssim: float = 0.0, sc_min_int: int = 1, sc_min_freq: int = 0, sc_normalize: bool = False,
                     sc_tht_white: float = constants.DEF_THT_WHITE, sc_tht_black: float = constants.DEF_THT_BLACK,
                     sc_debug: bool = False) -> vs.VideoNode:
    """Utility function to set the scene-change frames in the clip
    :param clip:                clip to process, any format is supported.
    :param sc_threshold:        Scene change threshold used to generate the reference frames.
                                It is a percentage of the luma change between the previous n-frame (n=sc_tht_offset)
                                and the current frame. range [0-1], default 0.10.
    :param sc_tht_offset:       Offset index used for the Scene change detection. The comparison will be performed,
                                between frame[n] and frame[n-offset]. An offset > 1 is useful to detect blended scene
                                change, range[1, 25]. Default = 1.
    :param sc_normalize:        If true the B&W frames are normalized before use misc.SCDetect(), the normalization will
                                increase the sensitivity to smooth scene changes.
    :param sc_tht_white:        Threshold to identify white frames, range [0-1], default 0.85.
    :param sc_tht_black:        Threshold to identify dark frames, range [0-1], default 0.15.
    :param sc_tht_ssim:         Threshold used by the SSIM (Structural Similarity Index Metric) selection filter.
                                If > 0, will be activated a filter that will improve the scene-change detection,
                                by discarding images that are similar.
                                Suggested values are between 0.35 and 0.85, range [0-1], default 0.0 (deactivated)
    :param sc_min_int:          Minimum number of frame interval between scene changes, range[1, 25]. Default = 1.
    :param sc_min_freq:         if > 0 will be generated at least a reference frame every "sc_min_freq" frames.
                                range [0-1500], default: 0.
    :param sc_debug:            Enable SC debug messages. default: False
    """
    clip, orig_fmt = convert_format_RGB24(clip)
    clip = SceneDetect(clip, threshold=sc_threshold, tht_offset=sc_tht_offset, frequency=sc_min_freq,
                       sc_tht_filter=sc_tht_ssim, min_length=sc_min_int, tht_white=sc_tht_white,
                       tht_black=sc_tht_black, frame_norm=sc_normalize, sc_debug=sc_debug)

    return restore_format(clip, orig_fmt)


def vs_SceneDetectEdges(clip: vs.VideoNode, sc_threshold: float = 0.035, sc_tht_offset: int = 2,
                     sc_tht_ssim: float = 0.80, sc_min_int: int = 20, sc_mult_tht: int = 15,
                     sc_tht_white: float = 0.70, sc_tht_black: float = 0.10,
                     sc_debug: bool = False) -> vs.VideoNode:
    """Utility function to set the scene-change frames in the clip
    :param clip:                clip to process, any format is supported.
    :param sc_threshold:        Scene change threshold used to generate the reference frames.
                                It is a percentage of the luma change between the previous n-frame edges
                                (n=sc_tht_offset) and the current frame edges. range [0.020-0.090], default 0.035.
    :param sc_tht_offset:       Offset index used for the Scene change detection. The comparison will be performed,
                                between frame[n] and frame[n-offset]. An offset > 1 is useful to detect blended scene
                                change, range[1, 25]. Default = 2.
    :param sc_tht_white:        Threshold to identify white frames, range [0-1], default 0.70.
    :param sc_tht_black:        Threshold to identify dark frames, range [0-1], default 0.10.
    :param sc_tht_ssim:         Threshold used by the SSIM (Structural Similarity Index Metric) selection filter.
                                If > 0, will be activated a filter that will improve the scene-change detection,
                                by discarding images that are similar.
                                Suggested values are between 0.35 and 0.85, range [0-1], default 0.80
    :param sc_min_int:          Minimum number of frame interval between scene changes, range[1, 25]. Default = 20.
    :param sc_mult_tht:         Threshold multiplier used to identify significant scene change, range[5, 25],
                                default = 15
    :param sc_debug:            Enable SC debug messages. default: False
    """
    clip, orig_fmt = convert_format_RGB24(clip)
    clip = SceneDetectEdges(clip, threshold=sc_threshold, ssim_threshold=sc_tht_ssim, sc_diff_offset=sc_tht_offset,
                            sc_min_int=sc_min_int, sc_mult_tht=sc_mult_tht, tht_white=sc_tht_white,
                            tht_black=sc_tht_black, sc_debug=sc_debug)

    return restore_format(clip, orig_fmt)

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
function to read a video clip
"""

def vs_read_video(source: str, fpsnum: int = 0, fpsden: int = 1, width: int = 0, height: int = 0,
                    return_rgb: bool = True) -> vs.VideoNode:
    """CMNET2 utility function to read a video provided externally.
       The clip provided in output will be already in RGB24 format

    :param source:       Full path to the video to read
    :param fpsnum:       FPS numerator, for using it in CMNET2, must be provided the
                         same value of clip to be colored: clip.fps_num
    :param fpsden:       FPS denominator, for using it in CMNET2, must be provided the
                         same value of clip to be colored: clip.fps_den
    :param width:        If > 0 the clip is resized to width using Spline36
    :param height:       If > 0 the clip is resized to height using Spline36
    :param return_rgb:   If True (default) the clip will be converted in RGB24 format
    :return:             clip in RGB24 format if return_rgb=True
    """
    if not os.path.isfile(source):
        CMNET2_LogMessage(MessageType.EXCEPTION, "CMNET2: invalid clip -> " + source)

    ext = source.lower()
    if not any(ext.endswith(extension) for extension in VIDEO_EXTENSIONS):
        CMNET2_LogMessage(MessageType.EXCEPTION, "CMNET2: invalid clip extension -> " + source)

    load_LSMASHSource_plugin()
    try:
        clip = vs.core.lsmas.LWLibavSource(source=source, stream_index=0, fpsnum=fpsnum, fpsden=fpsden,
                                           cache=0, prefer_hw=0)
    except Exception as error:
        clip = None
        CMNET2_LogMessage(MessageType.EXCEPTION, "CMNET2: LSMASHSource.dll not loaded or invalid clip -> " + str(error))

    if width > 0 and height > 0:
        clip = clip.resize.Spline36(width=width, height=height)
    elif width > 0:
        clip = clip.resize.Spline36(width=width, height=clip.height)
    elif height > 0:
        clip = clip.resize.Spline36(width=clip.width, height=height)

    # setting color matrix to 709.
    if cmnet2_utils._matrixIsInvalid(clip):
        clip = vs.core.std.SetFrameProps(clip, _Matrix=vs.MATRIX_BT709)
    # setting color transfer (vs.TRANSFER_BT709), if it is not set.
    if cmnet2_utils._transferIsInvalid(clip):
        clip = vs.core.std.SetFrameProps(clip=clip, _Transfer=vs.TRANSFER_BT709)
    # setting color primaries info (to vs.PRIMARIES_BT709), if it is not set.
    if cmnet2_utils._primariesIsInvalid(clip):
        clip = vs.core.std.SetFrameProps(clip=clip, _Primaries=vs.PRIMARIES_BT709)

    # setting color range to TV (limited) range.
    if vs.core.core_version.release_major < 74:
        clip = vs.core.std.SetFrameProps(clip=clip, _ColorRange=vs.RANGE_LIMITED)
    else:
        clip = vs.core.std.SetFrameProps(clip=clip, _Range=vs.RANGE_LIMITED)
    # making sure frame rate is set
    clip = vs.core.std.AssumeFPS(clip=clip, fpsnum=clip.fps_num, fpsden=clip.fps_den)
    # making sure the detected scan type is set (detected: progressive)
    clip = vs.core.std.SetFrameProps(clip=clip, _FieldBased=vs.FIELD_PROGRESSIVE)  # progressive
    if return_rgb:
        # adjusting color space to RGB24 for CMNET2
        matrix_str = cmnet2_utils._get_matrix_str(clip, default="709")
        clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s=matrix_str, range_s="full")
        # adjust _Matrix to RGB
        clip = vs.core.std.SetFrameProps(clip=clip, _Matrix=vs.MATRIX_RGB)
        # changing range from limited to full range for CMNET2
        clip = vs.core.resize.Bicubic(clip, range_in_s="limited", range_s="full")
        # setting color range to PC (full) range.
        if vs.core.core_version.release_major < 74:
            clip = vs.core.std.SetFrameProps(clip=clip, _ColorRange=vs.RANGE_FULL)
        else:
            clip = vs.core.std.SetFrameProps(clip=clip, _Range=vs.RANGE_FULL)
    else:
        # setting color range to TV (limited) range.
        if vs.core.core_version.release_major < 74:
            clip = vs.core.std.SetFrameProps(clip=clip, _ColorRange=vs.RANGE_LIMITED)
        else:
            clip = vs.core.std.SetFrameProps(clip=clip, _Range=vs.RANGE_LIMITED)

    return clip



"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
wrapper to function SceneDetect and vs_sc_export_frames() to export the clip's 
reference frames
"""


def vs_extract_reference_frames(clip: vs.VideoNode, sc_threshold: float = constants.DEF_THRESHOLD, sc_tht_offset: int = 1,
                                  sc_tht_ssim: float = 0.0, sc_min_int: int = 1, sc_min_freq: int = 0,
                                  sc_framedir: str = "./", sc_sequence: bool = False, sc_normalize: bool = False,
                                  ref_offset: int = 0, sc_tht_white: float = constants.DEF_THT_WHITE,
                                  sc_tht_black: float = constants.DEF_THT_BLACK, ref_ext: str = constants.DEF_EXPORT_FORMAT,
                                  ref_jpg_quality: int = constants.DEF_JPG_QUALITY, ref_override: bool = True,
                                  sc_algo:int =0, sc_debug: bool = False) -> vs.VideoNode:
    """Utility function to export reference frames
    :param clip:                clip to process, any format is supported.
    :param sc_threshold:        Scene change threshold used to generate the reference frames.
                                It is a percentage of the luma change between the previous n-frame (n=sc_offset)
                                and the current frame.
                                algo=0: suggested values are between 0.09 and 0.15 (best=0.10).
                                algo=1: suggested values are between 0.06 and 0.09 (best=0.07).
                                range [0-1], default 0.10.
    :param sc_tht_offset:       Offset index used for the Scene change detection. The comparison will be performed,
                                between frame[n] and frame[n-offset]. An offset > 1 is useful to detect blended scene
                                change.
                                algo=0: suggested values are between 1 and 5 (best=1).
                                algo=1: suggested values are between 1 and 5 (best=2).
                                range[1, 25]. Default = 1.
    :param sc_normalize:        If true the B&W frames are normalized before use misc.SCDetect(), the normalization will
                                increase the sensitivity to smooth scene changes (used only by algo=0).
    :param sc_tht_white:        Threshold to identify white frames, range [0-1], default 0.70.
    :param sc_tht_black:        Threshold to identify dark frames, range [0-1], default 0.10.
    :param sc_tht_ssim:         Threshold used by the SSIM (Structural Similarity Index Metric) selection filter.
                                If > 0, will be activated a filter that will improve the scene-change detection,
                                by discarding images that are similar.
                                algo=0: suggested values are between 0.35 and 0.85 (best=0.60).
                                algo=1: suggested values are between 0.10 and 0.20 (best=0.14).
                                range [0-1], default 0.0 (deactivated)
    :param sc_min_int:          Minimum number of frame interval between scene changes, range[1, 50]. Default = 1.
                                algo=0: suggested values are between 1 and 0.25 (best=10).
                                algo=1: suggested values are between 10 and 50 (best=30).
    :param sc_min_freq:         if > 0 will be generated at least a reference frame every "sc_min_freq" frames.
                                range [0-1500], default: 0 (auto).
                                algo=0: suggested values are between 0 and 25 (best=0).
                                algo=1: suggested values are between 5 and 10 (best=7).
    :param sc_framedir:         If set, define the directory where are stored the reference frames.
                                The reference frames are named as: ref_nnnnnn.[jpg|png].
    :param sc_sequence:         If True, the reference frames will be exported in sequence, using consecutive numbers.
    :param ref_offset:          Offset number that will be added to the number of generated frames. default: 0.
    :param ref_ext:             File extension and format of saved frames, range ["jpg", "png"] . default: "jpg"
    :param ref_jpg_quality:     Quality of "jpg" compression, range[0,100]. default: 95
    :param ref_override:        If True, the reference frames with the same name will be overridden, otherwise will
                                be discarded. default: True
    :param sc_algo:             Algorithm applied for scene detection, allowed values are:
                                   0: It will be applied standard SCDetect() method + SSIM detection
                                   1: It will be applied advance detection on the edges.
                                   2: It will be applied SCXvid plugin (very simple)
                                   3: IT will be used SCDetection from MVTools
                                Default = 0
    :param sc_debug:            Enable SC debug messages. default: False
    """
    clip, orig_fmt = convert_format_RGB24(clip)
    pathlib.Path(sc_framedir).mkdir(parents=True, exist_ok=True)
    if sc_algo == 0:
        clip = SceneDetect(clip, threshold=sc_threshold, tht_offset=sc_tht_offset, frequency=sc_min_freq,
                           sc_tht_filter=sc_tht_ssim, min_length=sc_min_int, tht_white=sc_tht_white,
                           tht_black=sc_tht_black, frame_norm=sc_normalize, sc_debug=sc_debug)
    else:
        clip = SceneDetectEdges(clip, threshold=sc_threshold, ssim_threshold=sc_tht_ssim, sc_diff_offset=sc_tht_offset,
                                sc_min_int=sc_min_int, sc_mult_tht=sc_min_freq, tht_white=sc_tht_white,
                                tht_black=sc_tht_black, sc_debug=sc_debug)

    clip = vs_sc_export_frames(clip, sc_framedir=sc_framedir, ref_offset=ref_offset, ref_ext=ref_ext,
                               ref_jpg_quality=ref_jpg_quality, ref_override=ref_override, sequence=sc_sequence)

    return restore_format(clip, orig_fmt)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
wrapper to function vs_sc_export_frames() to export the clip's reference frames
"""


def vs_export_reference_frames(clip: vs.VideoNode, sc_framedir: str = "./", ref_offset: int = 0,
                                 ref_ext: str = constants.DEF_EXPORT_FORMAT, ref_jpg_quality: int = constants.DEF_JPG_QUALITY,
                                 ref_override: bool = True) -> vs.VideoNode:
    """Utility function to export reference frames
    :param clip:                clip to process, any format is supported.
    :param sc_framedir:         If set, define the directory where are stored the reference frames.
                                The reference frames are named as: ref_nnnnnn.[jpg|png].
    :param ref_offset:          Offset number that will be added to the number of generated frames. default: 0.
    :param ref_ext:             File extension and format of saved frames, range ["jpg", "png"] . default: "jpg"
    :param ref_jpg_quality:     Quality of "jpg" compression, range[0,100]. default: 95
    :param ref_override:        If True, the reference frames with the same name will be overridden, otherwise will
                                be discarded. default: True
    """
    clip, orig_fmt = convert_format_RGB24(clip)
    pathlib.Path(sc_framedir).mkdir(parents=True, exist_ok=True)
    clip = vs_sc_export_frames(clip, sc_framedir=sc_framedir, ref_offset=ref_offset, ref_ext=ref_ext,
                               ref_jpg_quality=ref_jpg_quality, ref_override=ref_override)

    return restore_format(clip, orig_fmt)

def vs_export_list_frames(clip: vs.VideoNode, sc_framedir: str = "./", ref_list: list[int] | None= None,
                            offset: int = 0, ref_ext: str = constants.DEF_EXPORT_FORMAT, ref_jpg_quality: int = constants.DEF_JPG_QUALITY,
                            ref_override: bool = True, fast_extract: bool = True) -> vs.VideoNode:
    """Utility function to export frames included in a list.
    :param clip:                clip to process, any format is supported.
    :param sc_framedir:         If set, define the directory where are stored the frames.
                                The frames are named as: ref_nnnnnn.[jpg|png].
    :param ref_list:            List of frame numbers to export. default: None. If ref_list contains only one frame
                                number, for example ref_list = [25], will be exported a frame every 25 frames
    :param offset:              The offset will be added to the frame number. default = 0.
    :param ref_ext:             File extension and format of saved frames, range ["jpg", "png"] . default: "jpg"
    :param ref_jpg_quality:     Quality of "jpg" compression, range[0,100]. default: 95
    :param ref_override:        If True, the frames with the same name will be overridden, otherwise will
                                be discarded. default: True
    :param fast_extract:        If True, the frames will be extracted directly with get_frame(), otherwise will
                                be performed a full parsing of the clip (necessary if there is a sequential temporal
                                dependency in the script calling this function). default = True
    """
    if ref_list is None or len(ref_list) < 1:
        return clip

    clip, orig_fmt = convert_format_RGB24(clip)
    pathlib.Path(sc_framedir).mkdir(parents=True, exist_ok=True)
    clip = vs_list_export_frames(clip, sc_framedir=sc_framedir, ref_list=ref_list, ref_ext=ref_ext, offset=offset,
                                 ref_jpg_quality=ref_jpg_quality, ref_override=ref_override,fast_extract=fast_extract)

    return restore_format(clip, orig_fmt)

"""
------------------------------------------------------------------------------------------------------------------------ 
                                   CMNET2 INTERNAL FUNCTIONS
------------------------------------------------------------------------------------------------------------------------ 
"""

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
wrapper to function vs_sc_export_frames() to export the clip's reference frames
"""


def _extract_reference_frames(clip: vs.VideoNode, sc_framedir: str = "./", ref_offset: int = 0, ref_ext: str = "png",
                              ref_override: bool = True, prop_name: str = "_SceneChangePrev") -> vs.VideoNode:
    """Export scene-change frames from clip to a directory on disk.
    Creates sc_framedir if it does not exist, converts the clip to RGB24, then delegates
    to vs_sc_export_frames. Primarily used by HAVC_main to persist reference frames for
    subsequent exemplar-based passes.
    :param clip:         Input clip (any format).
    :param sc_framedir:  Output directory for reference images. Created if missing.
    :param ref_offset:   Value added to the frame number in each filename. Default 0.
    :param ref_ext:      Image format extension ('png' or 'jpg'). Default 'png'.
    :param ref_override: If False, existing files are not overwritten. Default True.
    :param prop_name:    Frame property used to detect scene changes. Default '_SceneChangePrev'.
    :return:             Clip pass-through (side-effect: reference frames saved to disk).
    """
    pathlib.Path(sc_framedir).mkdir(parents=True, exist_ok=True)
    clip, orig_fmt = convert_format_RGB24(clip)
    clip = vs_sc_export_frames(clip, sc_framedir=sc_framedir, ref_offset=ref_offset, ref_ext=ref_ext,
                               ref_override=ref_override, prop_name=prop_name)
    return restore_format(clip, orig_fmt)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
wrapper to function vs_recover_clip_luma().
"""


def _clip_chroma_resize(clip_hires: vs.VideoNode, clip_lowres: vs.VideoNode) -> vs.VideoNode:
    """Upscale clip_lowres to the dimensions of clip_hires and replace its luma with the hi-res luma.
    :param clip_hires:  High-resolution clip whose luma plane will be preserved.
    :param clip_lowres: Low-resolution clip whose chroma planes will be upscaled and blended in.
    :return:            Clip at clip_hires resolution with luma from clip_hires and chroma from the upscaled clip_lowres.
    """
    clip_resized = clip_lowres.resize.Spline64(width=clip_hires.width, height=clip_hires.height)
    clip_hires, orig_fmt_h = convert_format_RGB24(clip_hires)
    clip_resized, orig_fmt_r = convert_format_RGB24(clip_resized)
    clip_recovered = vs_recover_clip_luma(clip_hires, clip_resized)
    return restore_format(clip_recovered, orig_fmt_h)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
wrapper to function vs_get_clip_frame() to get frames fast.
"""


def _get_clip_frame(clip: vs.VideoNode, nframe: int = 0) -> vs.VideoNode:
    """Extract a single frame from the clip and return it as a one-frame clip, preserving the original format.
    :param clip:    Input clip, any format is supported.
    :param nframe:  Zero-based index of the frame to extract. Default = 0.
    :return:        Single-frame clip containing the requested frame in the original clip format.
    """
    clip, orig_fmt = convert_format_RGB24(clip)
    clip = vs_get_clip_frame(clip=clip, nframe=nframe)
    return restore_format(clip, orig_fmt)


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
disable packages warnings.
"""


def disable_warnings():
    """Suppress noisy log output from third-party libraries used by CMNET2.
    Sets the log level to ERROR for known verbose packages (matplotlib, PIL, torch,
    numpy, tensorrt, kornia, dinov2) and silences FutureWarning, UserWarning, and
    DeprecationWarning categories project-wide.
    """
    logger_blocklist = [
        "matplotlib",
        "PIL",
        "torch",
        "numpy",
        "tensorrt",
        "torch_tensorrt"
        "kornia",
        "dinov2"  # dinov2 is issuing warnings not allowing ColorMNetServer to work properly
    ]
    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.ERROR)

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    # warnings.simplefilter(action="ignore", category=Warning)
    torch._logging.set_logs(all=logging.ERROR)
