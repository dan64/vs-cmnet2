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
CMNET2 utility filter functions
"""
from __future__ import annotations
import vapoursynth as vs
from functools import partial
import os
import math
import re
import cv2
from PIL import Image
import numpy as np
from typing import NamedTuple


from .vsslib import restcolor as restcolor
from .vsslib import vsresize as vsresize
from .vsslib.constants import *
from .vsslib.vsutils import CMNET2_LogMessage, MessageType, frame_to_image, image_to_frame


# Using NamedTuple for better compatibility with VapourSynth's typical usage patterns
class ClipInfo(NamedTuple):
    clip_orig: vs.VideoNode | None
    format_id: int
    color_family: vs.ColorFamily
    bits_per_sample: int
    matrix: int | None
    color_range: int | None
    chroma_resize: bool
    # luma protect fields
    clip_high_bitdepth: vs.VideoNode | None = None  # YUV444PS reference for luma restoration
    preserve_luma: bool = False
    luma_blend: float = 1.0  # 1.0 = 100% original luma, 0.0 = 100% CMNET2 luma


VIDEO_EXTENSIONS = ['.mpg', '.mp4', '.m4v', '.avi', '.mkv', '.mpeg']

# map integer _Matrix to zimg string
MATRIX_INT_TO_STR = {
    0: "rgb",
    1: "709",
    4: "fcc",
    5: "470bg",
    6: "170m",
    7: "240m",
    8: "ycgco",
    9: "2020ncl",
    10: "2020cl",
}

# bit depth threshold for automatic luma preservation
LUMA_PROTECT_MIN_BITDEPTH = 10


def _get_matrix_str(clip: vs.VideoNode, default: str = "709") -> str:
    """Read _Matrix from frame props and return the equivalent zimg string."""
    if _matrixIsInvalid(clip):
        return default
    matrix_val = clip.get_frame(0).props.get('_Matrix')
    return MATRIX_INT_TO_STR.get(int(matrix_val), default)


def _matrix_int_to_str(matrix_val, default: str = "709") -> str:
    """Convert an integer/enum _Matrix value to the zimg string."""
    if matrix_val is None:
        return default
    return MATRIX_INT_TO_STR.get(int(matrix_val), default)


def _should_preserve_luma(clip: vs.VideoNode, preserve_luma: bool | None) -> bool:
    """
    Decide whether to enable luma preservation.
    - If preserve_luma is explicitly True/False, honor it.
    - If None (auto), enable only for high bit-depth sources (>= 10-bit) that are YUV/RGB.
    """
    if preserve_luma is not None:
        return preserve_luma
    fmt = clip.format
    if fmt is None:
        return False
    if fmt.bits_per_sample < LUMA_PROTECT_MIN_BITDEPTH:
        return False
    # GRAY: no chroma to colorize-and-merge, luma protect is meaningless
    if fmt.color_family == vs.GRAY:
        return False
    return True


def _build_high_bitdepth_reference(clip: vs.VideoNode) -> vs.VideoNode | None:
    """
    Build a YUV444PS reference clip from the original high-bitdepth input.
    Used later to extract the original Y plane in restore_format().
    Returns None if the input can't be converted.
    """
    fmt = clip.format
    if fmt is None:
        return None

    if fmt.color_family == vs.YUV:
        # YUV → YUV444PS: chroma upsampling + float promotion.
        # No matrix conversion needed (stays YUV).
        return vs.core.resize.Bicubic(clip, format=vs.YUV444PS)

    if fmt.color_family == vs.RGB:
        # RGB → YUV444PS: needs a matrix. Use BT.709 as a conventional choice
        # for HD content; this matrix is only used to derive Y for protection.
        return vs.core.resize.Bicubic(
            clip,
            format=vs.YUV444PS,
            matrix_s="709",
            range_in_s="full",
            range_s="limited",
        )

    return None


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
function to convert video clip to RGB24 format and to restore the original format
"""

def convert_format_RGB24(
    clip: vs.VideoNode,
    chroma_resize: bool = False,
    preserve_luma: bool | None = None,
    luma_blend: float = 1.0,
) -> tuple[vs.VideoNode, ClipInfo]:
    """
    Convert any clip to RGB24 (8-bit full-range RGB).
    :param clip:           input clip (YUV/RGB/GRAY, any bit depth)
    :param chroma_resize:  if True, resize to a chroma-friendly size for CMNET2
    :param preserve_luma:  if True, keep a YUV444PS copy of the source so the
                           original high-bitdepth luma can be restored in
                           restore_format(). If None (default), auto-enable
                           for sources with bit depth >= 10.
    :param luma_blend:     blend factor for luma restoration in [0.0, 1.0].
                           1.0 = use 100% original luma (default),
                           0.0 = use 100% CMNET2-produced luma.
    :return: (rgb24_clip, ClipInfo)
    """
    # Store original clip information before any processing
    original_format = clip.format
    if original_format is None:
        CMNET2_LogMessage(MessageType.EXCEPTION, "Clip must have a defined format")

    if not isinstance(clip, vs.VideoNode):
        CMNET2_LogMessage(MessageType.EXCEPTION, "convert_format_RGB24: Input is not a valid clip.")

    # Decide luma protect and build the reference clip if needed
    do_preserve_luma = _should_preserve_luma(clip, preserve_luma)
    high_bd_clip = _build_high_bitdepth_reference(clip) if do_preserve_luma else None
    # If reference build failed, disable luma protect rather than crash later
    if do_preserve_luma and high_bd_clip is None:
        do_preserve_luma = False
    # Clamp blend factor
    luma_blend = max(0.0, min(1.0, float(luma_blend)))
    # Fast path: already RGB24, assume CMNET2_read_video produced it
    if clip.format.id == vs.RGB24:
        if vs.core.core_version.release_major < 74:
            clip_color_range = vs.ColorRange(vs.RANGE_FULL)
        else:
            clip_color_range = vs.Range(vs.RANGE_FULL)

        clip_info = ClipInfo(
            clip_orig=clip if chroma_resize else None,
            format_id=original_format.id,
            color_family=original_format.color_family,
            bits_per_sample=original_format.bits_per_sample,
            matrix=vs.MatrixCoefficients(vs.MATRIX_RGB),
            color_range=clip_color_range,
            chroma_resize=chroma_resize,
            clip_high_bitdepth=high_bd_clip,
            preserve_luma=do_preserve_luma,
            luma_blend=luma_blend,
        )
        if chroma_resize:
            clip = vsresize.resize_min_HW(clip)
        return clip, clip_info

    # Not RGB24: normalize missing color metadata to sane defaults
    if _matrixIsInvalid(clip):
        clip = clip.std.SetFrameProps(_Matrix=vs.MATRIX_BT709)
    if _rangeIsInvalid(clip):
        if vs.core.core_version.release_major < 74:
            clip = clip.std.SetFrameProps(_ColorRange=vs.RANGE_LIMITED)
        else:
            clip = clip.std.SetFrameProps(_Range=vs.RANGE_LIMITED)

    # Read props after normalization, so ClipInfo reflects the actual values
    frame = clip.get_frame(0)
    props = frame.props
    if vs.core.core_version.release_major < 74:
        clip_color_range = vs.ColorRange(props.get('_ColorRange', vs.RANGE_LIMITED.value))
    else:
        clip_color_range = vs.Range(props.get('_Range', vs.RANGE_LIMITED.value))

    clip_info = ClipInfo(
        clip_orig=clip if chroma_resize else None,
        format_id=original_format.id,
        color_family=original_format.color_family,
        bits_per_sample=original_format.bits_per_sample,
        matrix=vs.MatrixCoefficients(props.get('_Matrix', vs.MATRIX_BT709.value)),
        color_range=clip_color_range,
        chroma_resize=chroma_resize,
        clip_high_bitdepth=high_bd_clip,
        preserve_luma=do_preserve_luma,
        luma_blend=luma_blend,
    )
    if chroma_resize:
        clip = vsresize.resize_min_HW(clip)

    # Ensure we're working with 8-bit before the RGB conversion
    if clip.format.bits_per_sample != 8:
        clip = vs.core.resize.Bicubic(clip, format=clip.format.replace(bits_per_sample=8))

    # Convert to RGB24 based on the original color family
    if original_format.color_family == vs.YUV:
        matrix_val = int(clip.get_frame(0).props.get('_Matrix', 1))
        matrix_str = MATRIX_INT_TO_STR.get(matrix_val, "709")
        clip = vs.core.resize.Bicubic(
            clip,
            format=vs.RGB24,
            matrix_in_s=matrix_str,
            range_in_s="limited",
            range_s="full",
            dither_type="error_diffusion",
        )
    elif original_format.color_family == vs.GRAY:
        clip = vs.core.resize.Bicubic(
            clip,
            format=vs.RGB24,
            range_in_s="limited",
            range_s="full",
        )
    else:  # Already RGB but not RGB24 (e.g., RGB48, RGBS)
        clip = vs.core.resize.Bicubic(
            clip,
            format=vs.RGB24,
            range_s="full",
        )

    # After conversion to RGB, _Matrix must be RGB (0)
    clip = clip.std.SetFrameProps(_Matrix=vs.MATRIX_RGB)
    # Mark output as full-range RGB
    if vs.core.core_version.release_major < 74:
        clip = clip.std.SetFrameProps(_ColorRange=vs.RANGE_FULL)
    else:
        clip = clip.std.SetFrameProps(_Range=vs.RANGE_FULL)

    return clip, clip_info


def _restore_with_luma_protect(
    clip: vs.VideoNode,
    clip_info: ClipInfo,
    target_format_id: int | None = None,
) -> vs.VideoNode:
    """
    Convert the CMNET2 RGB24 output to YUV444PS, replace (or blend) the Y plane
    with the original high-bitdepth luma, then convert to the target format.
    :param clip:              CMNET2-processed clip in RGB24 full-range
    :param clip_info:         ClipInfo from convert_format_RGB24
    :param target_format_id:  desired output format; defaults to the original
                              format stored in clip_info
    """
    if clip_info.clip_high_bitdepth is None:
        CMNET2_LogMessage(
            MessageType.EXCEPTION,
            "restore_format: preserve_luma is set but no high-bitdepth reference is available.",
        )

    if target_format_id is None:
        target_format_id = clip_info.format_id

    # 1. Convert CMNET2 RGB24 output to YUV444PS (float, 4:4:4)
    #    The matrix used here must match the matrix the original source carried,
    #    so the merged luma stays semantically coherent with the new chroma.
    matrix_str = _matrix_int_to_str(clip_info.matrix, default="709")
    # Original color range determines the YUV target range
    if clip_info.color_range is not None and clip_info.color_range == vs.RANGE_FULL:
        yuv_range_s = "full"
    else:
        yuv_range_s = "limited"

    havc_yuv = vs.core.resize.Bicubic(
        clip,
        format=vs.YUV444PS,
        matrix_s=matrix_str,
        range_in_s="full",
        range_s=yuv_range_s,
        dither_type="error_diffusion",
    )
    # 2. Extract Y from CMNET2 output and from the original high-bitdepth reference
    havc_y = vs.core.std.ShufflePlanes(havc_yuv, planes=0, colorfamily=vs.GRAY)
    orig_y = vs.core.std.ShufflePlanes(
        clip_info.clip_high_bitdepth, planes=0, colorfamily=vs.GRAY
    )
    # 3. Build the protected Y plane
    blend = clip_info.luma_blend
    if blend >= 1.0:
        protected_y = orig_y
    elif blend <= 0.0:
        protected_y = havc_y
    else:
        # weighted average: protected = blend * orig + (1 - blend) * havc
        expr = f"x {blend} * y {1.0 - blend} * +"
        protected_y = vs.core.std.Expr([orig_y, havc_y], expr=expr)

    # 4. Recombine planes: protected Y + CMNET2 U/V
    merged = vs.core.std.ShufflePlanes(
        clips=[protected_y, havc_yuv, havc_yuv],
        planes=[0, 1, 2],
        colorfamily=vs.YUV,
    )
    # 5. Convert to the requested target format
    if merged.format.id == target_format_id:
        return merged

    return vs.core.resize.Bicubic(
        merged,
        format=target_format_id,
        dither_type="error_diffusion",
    )


def restore_format(
    clip: vs.VideoNode,
    clip_info: ClipInfo,
    target_format_id: int | None = None,
) -> vs.VideoNode:
    """
    Restore the colorized RGB24 clip to a format suitable for the original input.
    - If original was GRAY, output YUV420P8 (8-bit color).
    - If original was YUV, restore to original YUV format.
    - If original was RGB, restore to original RGB format.
    Assumes input 'clip' is full-range RGB24.
    :param clip:              clip to process (must be RGB24 full-range).
    :param clip_info:         ClipInfo struct containing original clip information.
    :param target_format_id:  optional override of the output format. If None,
                              uses the original format from clip_info. Useful for
                              keeping the result in YUV444PS for further processing.
    """
    if not isinstance(clip, vs.VideoNode):
        CMNET2_LogMessage(MessageType.EXCEPTION, "restore_format: Input is not a valid clip.")

    if clip.format.id != vs.RGB24:
        CMNET2_LogMessage(MessageType.EXCEPTION, "restore_format: Input clip must be RGB24.")

    if clip_info.chroma_resize:
        clip = vsresize.resize_to_chroma(clip_info.clip_orig, clip)

    # Luma-protect path: preserves original high-bitdepth Y
    if clip_info.preserve_luma:
        return _restore_with_luma_protect(clip, clip_info, target_format_id)

    # Standard path (8-bit round-trip)
    output_format_id = target_format_id if target_format_id is not None else clip_info.format_id
    # If already in target format (unlikely post-colorization), return as-is
    if clip.format.id == output_format_id:
        return clip

    if clip_info.color_family == vs.YUV:
        matrix = clip_info.matrix if clip_info.matrix is not None else vs.MATRIX_BT709
        range_s = "limited"
        if clip_info.color_range is not None:
            range_s = "full" if clip_info.color_range == vs.RANGE_FULL else "limited"

        restored = vs.core.resize.Bicubic(
            clip,
            format=output_format_id,
            matrix_in=vs.MATRIX_RGB,
            matrix=matrix,
            range_in_s="full",
            range_s=range_s,
            dither_type="error_diffusion",
        )
    elif clip_info.color_family == vs.GRAY:
        # Original was grayscale → output 8-bit YUV (colorized result)
        range_s = "limited"
        if clip_info.color_range is not None:
            range_s = "full" if clip_info.color_range == vs.RANGE_FULL else "limited"

        # If target_format_id was overridden to something compatible, honor it;
        # otherwise default to YUV420P8 (consumer-friendly colorized output).
        gray_target = output_format_id if target_format_id is not None else vs.YUV420P8
        restored = vs.core.resize.Bicubic(
            clip,
            format=gray_target,
            matrix=vs.MATRIX_BT709,
            range_in_s="full",
            range_s=range_s,
            dither_type="error_diffusion",
        )
    else:
        # Original was RGB (but not RGB24, e.g., RGB48, RGBS)
        range_s = "full"
        if clip_info.color_range is not None:
            range_s = "full" if clip_info.color_range == vs.RANGE_FULL else "limited"

        restored = vs.core.resize.Bicubic(
            clip,
            format=output_format_id,
            range_in_s="full",
            range_s=range_s,
        )

    return restored


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: 
------------------------------------------------------------------------------- 
Utility functions for CMNET2_main
"""


def _get_render_factors(Preset: str) -> tuple[int, int, int]:
    # Select presets / tuning
    Preset = Preset.lower()
    presets = ['placebo', 'veryslow', 'slower', 'slow', 'medium', 'fast', 'faster', 'veryfast']
    preset0_rf = [36, 34, 32, 28, 24, 22, 20, 16]
    preset1_rf = [36, 34, 32, 28, 24, 22, 20, 16]
    pr_id = 5  # default 'fast'
    try:
        pr_id = presets.index(Preset)
    except ValueError:
        CMNET2_LogMessage(MessageType.EXCEPTION, "CMNET2_main: Preset choice is invalid for '" + str(pr_id) + "'")

    return pr_id, preset0_rf[pr_id], preset1_rf[pr_id]


def _get_mweight(VideoTune: str) -> float:
    # Select VideoTune
    VideoTune = VideoTune.lower()
    video_tune = ['verystable', 'morestable', 'stable', 'balanced', 'vivid', 'morevivid', 'veryvivid']
    ddcolor_weight = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    w_id = 3
    try:
        w_id = video_tune.index(VideoTune)
    except ValueError:
        CMNET2_LogMessage(MessageType.EXCEPTION, "CMNET2_main: VideoTune choice is invalid for '" + VideoTune + "'")

    return ddcolor_weight[w_id]


def _get_comb_method(CombMethod: str) -> int:
    # Select VideoTune
    CombMethod = CombMethod.lower()
    comb_str = ['simple', 'constrained-chroma', 'luma-masked', 'adaptive-luma', 'chroma-retention', 'chromabound adaptive']
    method_id = [2, 3, 4, 5, 6, 7]
    w_id = 2
    try:
        w_id = comb_str.index(CombMethod)
    except ValueError:
        CMNET2_LogMessage(MessageType.EXCEPTION, "CMNET2_main: CombMethod choice is invalid for '" + CombMethod + "'")

    return method_id[w_id]

def _spit_color_model(ColorModel: str) -> tuple[str, str]:
    ColorModel = ColorModel.lower()
    deoldify = "none"
    ddcolor = "none"
    if '+' not in ColorModel:
        if 'deoldify' in ColorModel:
            return ColorModel, ddcolor
        else:
            return deoldify, ColorModel

    cm = ColorModel.split("+")
    deoldify = f"deoldify({cm[0]})"
    if cm[1] in ('siggraph17', 'eccv16'):
        ddcolor = f"zhang({cm[1]})"
    else:
        ddcolor = f"ddcolor({cm[1]})"

    return deoldify, ddcolor

def _get_color_model(ColorModel: str) -> tuple[int, int, int]:
    ColorModel = ColorModel.lower()
    ddcolor_list = ["modelscope", "artistic", "siggraph17", "eccv16"]
    deoldify_list = ["video", "stable", "artistic"]
    if '+' in ColorModel:
        cm = ColorModel.split("+")
        do_model = deoldify_list.index(cm[0])
        dd_model = ddcolor_list.index(cm[1])
        dd_method = 2
        return do_model, dd_model, dd_method

    do_model = 0  # default: Video
    dd_model = 0  # default: Modelscope
    cmodel = ""
    if "deoldify" in ColorModel:
        cmodel = ColorModel.replace("deoldify", "").replace("(", "").replace(")", "")
        do_model = deoldify_list.index(cmodel)
        dd_method = 0
        return do_model, dd_model, dd_method

    if "ddcolor" in ColorModel:
        cmodel = ColorModel.replace("ddcolor", "").replace("(", "").replace(")", "")
    elif "zhang" in ColorModel:
        cmodel = ColorModel.replace("zhang", "").replace("(", "").replace(")", "")
    else:
        CMNET2_LogMessage(MessageType.EXCEPTION, "CMNET2_main: ColorModel choice is invalid for '" + ColorModel + "'")

    dd_method = 1
    dd_model = ddcolor_list.index(cmodel)
    return do_model, dd_model, dd_method


def _get_temp_color(ColorTemp: str) -> int:
    # Select ColorTemp
    if ColorTemp is None:
        ColorTemp = "none"
    ColorTemp = ColorTemp.lower().replace(" ", "")
    color_temp = ['none', 'veryhigh', 'high', 'medium', 'low', 'verylow']
    ct_id = color_temp.index(ColorTemp)
    return ct_id


def _get_color_tune(ColorTune: str, ColorFix: str, ColorMap: str,
                    dd_model: int) -> tuple[list[bool], str, str, str, str]:
    # set defaults
    dd_tweak = [False, False, False]   # tweaks_enabled, denoise_enabled, retinex_enabled
    hue_range = ""
    hue_range2 = ""
    chroma_adjust = ""
    chroma_adjust2 = ""
    # Select ColorTune for ColorFix
    if ColorTune is None:
        ColorTune = "none"
    ColorTune = ColorTune.lower()
    color_tune = ['none', 'light', 'medium', 'strong']
    if dd_model == 0:
        hue_tune = ["1.0,0.0", "0.7,0.1", "0.5,0.1", "0.2,0.1"]
    elif dd_model == 2:
        hue_tune = ["1.0,0.0", "0.6,0.1", "0.4,0.2", "0.2,0.1"]
    elif dd_model == 3:
        hue_tune = ["1.0,0.0", "0.7,0.1", "0.6,0.1", "0.3,0.1"]
    else:
        hue_tune = ["1.0,0.0", "0.8,0.1", "0.5,0.1", "0.2,0.1"]
    hue_tune2 = ["1.0,0.0", "0.9,0", "0.7,0", "0.5,0"]
    tn_id = 0
    try:
        tn_id = color_tune.index(ColorTune)
    except ValueError:
        CMNET2_LogMessage(MessageType.EXCEPTION, "CMNET2_main: ColorTune choice is invalid for '" + ColorTune + "'")

    # Select ColorFix for ddcolor/stabilizer
    if ColorFix is None:
        ColorFix = "none"

    ColorFix = ColorFix.lower()
    color_fix = ['none', 'magenta', 'magenta/violet', 'violet', 'violet/red', 'blue/magenta', 'yellow', 'yellow/orange',
                 'yellow/green', 'retinex/red']
    hue_fix = ["none", "270:300", "250:360", "300:330", "300:360", "220:280", "60:90", "30:90", "60:120", "none"]
    co_id = 5
    try:
        co_id = color_fix.index(ColorFix)
    except ValueError:
        CMNET2_LogMessage(MessageType.EXCEPTION, "CMNET2_main: ColorFix choice is invalid for '" + ColorFix + "'")

    if tn_id == 0:
        hue_range = "none"
        hue_range2 = "none"
        # in this case all the Tweaks for DDcolor are disabled
        dd_tweak[0] = False
    elif tn_id != 0 and co_id == 0:
        hue_range = "none"
        hue_range2 = "none"
        # in this case the tweaks/denoise for DDcolor are enabled but hue adjust is disabled
        dd_tweak[0] = True
        dd_tweak[1] = True
    elif tn_id != 0 and co_id == 9:
        hue_range = hue_fix[4] +  "|" + hue_tune[2]
        hue_range2 = hue_fix[4] + "|" + hue_tune2[2]
        # in this case the tweaks/retinex for DDcolor are enabled and hue adjust is set to 'violet/red', tune = 'medium'
        dd_tweak[0] = True
        dd_tweak[2] = True
    else:
        hue_range = hue_fix[co_id] + "|" + hue_tune[tn_id]
        hue_range2 = hue_fix[co_id] + "|" + hue_tune2[tn_id]
        dd_tweak[0] = True  # in this case the Tweaks for DDcolor are enabled

    # Select Color Mapping
    ColorMap = ColorMap.lower()
    hue_w = ["1.0", "0.90", "0.80", "0.75"]
    colormap = ['none', 'blue->brown', 'blue->red', 'blue->green', 'green->brown', 'green->red', 'green->blue',
                'redrose->brown', 'redrose->blue', "red->brown", 'red->blue', 'yellow->rose']
    hue_map = ["none", "180:280|+140", "180:280|+100", "180:280|+220", "80:180|+260", "80:180|+220", "80:180|+140",
               "300:360,0:20|+40", "300:360,0:20|+260", "320:360|+50", "300:360|+260", "30:90|+300"]

    cl_id = 0
    try:
        cl_id = colormap.index(ColorMap)
    except ValueError:
        ret_range = restcolor._parse_hue_adjust(ColorMap)
        if ret_range is None:
            CMNET2_LogMessage(MessageType.EXCEPTION, "CMNET2_main: ColorMap choice is invalid for '" + ColorMap + "'")
        else:
            cl_id = -1

    if cl_id == 0:
        chroma_adjust = "none"
        chroma_adjust2 = "none"
    elif cl_id == -1:
        chroma_adjust = ColorMap
        chroma_adjust2 = "none"
    else:
        chroma_adjust = hue_map[cl_id] + "," + hue_w[tn_id]
        if tn_id == 0:
            chroma_adjust2 = "none"
        else:
            chroma_adjust2 = chroma_adjust

    return dd_tweak, hue_range, hue_range2, chroma_adjust, chroma_adjust2

def _get_colormap(ColorMap: str = "red->brown", ColorTune: str ="light") -> str:
    color_tune = ['none', 'light', 'medium', 'strong']
    tn_id = 0
    try:
        tn_id = color_tune.index(ColorTune)
    except ValueError:
        CMNET2_LogMessage(MessageType.EXCEPTION, "CMNET2_main: ColorTune choice is invalid for '" + ColorTune + "'")

    # Select Color Mapping
    ColorMap = ColorMap.lower()
    hue_w = ["1.0", "0.90", "0.80", "0.75"]
    colormap = ['none', 'blue->brown', 'blue->red', 'blue->green', 'green->brown', 'green->red', 'green->blue',
                'redrose->brown', 'redrose->blue', "red->brown", 'red->blue', 'yellow->rose']
    hue_map = ["none", "180:280|+140", "180:280|+100", "180:280|+220", "80:180|+260", "80:180|+220", "80:180|+140",
               "300:360,0:20|+40", "300:360,0:20|+260", "320:360|+50", "300:360|+260", "30:90|+300"]

    cl_id = 0
    try:
        cl_id = colormap.index(ColorMap)
    except ValueError:
        ret_range = restcolor._parse_hue_adjust(ColorMap)
        if ret_range is None:
            CMNET2_LogMessage(MessageType.EXCEPTION, "CMNET2_main: ColorMap choice is invalid for '" + ColorMap + "'")
        else:
            return ColorMap

    chroma_adjust = hue_map[cl_id] + "," + hue_w[tn_id]
    return chroma_adjust

def _get_tune_id(bw_tune: str) -> int:
    BWTune = bw_tune.lower()
    bw_tune_list = ['none', 'light', 'medium', 'strong']
    tn_id = bw_tune_list.index(BWTune)
    return tn_id

def _check_input(DeepExOnlyRefFrames: bool, ScFrameDir: str, DeepExMethod: int, ScThreshold: float,
                 ScMinFreq: int, DeepExRefMerge: int):
    if DeepExOnlyRefFrames and (ScFrameDir is None):
        CMNET2_LogMessage(MessageType.EXCEPTION, "CMNET2_main: DeepExOnlyRefFrames is enabled but ScFrameDir is unset")

    if not (ScFrameDir is None) and DeepExMethod != 0 and DeepExOnlyRefFrames:
        CMNET2_LogMessage(MessageType.EXCEPTION,
                        "CMNET2_main: DeepExOnlyRefFrames is enabled but method not = 0 (CMNET2)")

    if (DeepExMethod != 0 and DeepExMethod != DEF_CMNET2_METHOD_PLACEBO) and (ScFrameDir is None):
        CMNET2_LogMessage(MessageType.EXCEPTION, "CMNET2_main: DeepExMethod != 0 but ScFrameDir is unset")

    if DeepExMethod in (0, 1, 2, 5, 6, DEF_CMNET2_METHOD_PLACEBO) and ScThreshold == 0 and ScMinFreq == 0:
        CMNET2_LogMessage(MessageType.EXCEPTION,
                        "CMNET2_main: DeepExMethod in (0, 1, 2, 5, 6) but ScThreshold and ScMinFreq are not set")

    if DeepExMethod in (2, 6) and DeepExRefMerge > 0:
        CMNET2_LogMessage(MessageType.EXCEPTION,
                        "CMNET2_main: RefMerge cannot be used with DeepExMethod in (2, 6)")

def get_ref_number(filename) -> int | None:
    if filename is None:
        return None
    match = re.search(r'ref_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

# ------------------------------------------------------------
# collection of small helper functions to validate parameters
# ------------------------------------------------------------

def is_limited_range(clip: vs.VideoNode) -> bool:
    # Try to read _ColorRange/_Range from props without forcing frame eval if possible.
    # Unfortunately, VapourSynth doesn't expose props without get_frame(),
    # so we have to accept minimal frame access—but make it safe.
    try:
        props = clip.get_frame(0).props
        if vs.core.core_version.release_major < 74:
            color_range = props.get('_ColorRange', vs.RANGE_LIMITED)  # default to limited if missing
        else:
            color_range = props.get('_Range', vs.RANGE_LIMITED)
        return color_range == vs.RANGE_LIMITED
    except Exception:
        # Fallback: assume full range if frame access fails
        return False

def _matrixIsInvalid(clip: vs.VideoNode) -> bool:
    frame = clip.get_frame(0)
    value = frame.props.get('_Matrix', None)
    # Non specificato o riservato
    if value in (None, 2, 3):
        return True

    # Non un membro valido dell'enum
    if value not in vs.MatrixCoefficients.__members__.values():
        return True

    # Coerenza con il color family
    if clip.format.color_family == vs.RGB and value != 0:
        return True  # RGB deve avere _Matrix=0
    if clip.format.color_family in (vs.YUV, vs.GRAY) and value == 0:
        return True  # YUV/GRAY non può avere _Matrix=RGB

    return False

def _transferIsInvalid(clip: vs.VideoNode) -> bool:
    frame = clip.get_frame(0)
    value = frame.props.get('_Transfer', None)
    return value in [None, 0, 2, 3] or value not in vs.TransferCharacteristics.__members__.values()


def _primariesIsInvalid(clip: vs.VideoNode) -> bool:
    frame = clip.get_frame(0)
    value = frame.props.get('_Primaries', None)
    return value in [None, 2] or value not in vs.ColorPrimaries.__members__.values()


def _rangeIsInvalid(clip: vs.VideoNode) -> bool:
    frame = clip.get_frame(0)
    if vs.core.core_version.release_major < 74:
        value = frame.props.get('_ColorRange', None)
        return value is None or value not in vs.ColorRange.__members__.values()
    else:
        value = frame.props.get('_Range', None)
        return value is None or value not in vs.Range.__members__.values()

def _fieldBaseIsInvalid(clip: vs.VideoNode) -> bool:
    frame = clip.get_frame(0)
    value = frame.props.get('_FieldBased', None)
    return value is None or value not in vs.FieldBased.__members__.values()


def adjust_rgb(clip: vs.VideoNode, factor: list = (1.0, 1.0, 1.0), bias: list = (0, 0, 0),
               gamma: list = (1.0, 1.0, 1.0)) -> vs.VideoNode:
    """Utility function to change the color and luminance of RGB clip.
       Gain, bias (offset) and gamma can be set independently on each channel.
       :param clip:         Clip to process. Only RGB24 format is supported.
       :param factor:       List of Red, green and blue scaling factor, in the list format: (r, g, b).
                            Range 0.0 to 255.0, default = (1, 1, 1).
                            For example, r=1.3 multiplies the red channel pixel values by 1.3.
       :param bias:         List of Red, green and blue bias adjustments, in the list format: (rb, gb, bb).
                            Bias adjustment—add a fixed positive or negative value to a channel's pixel values.
                            For example, rb=16 will add 16 to all red pixel values and rb=-32 will subtract 32 from all
                            red pixel values, default = (0, 0, 0).
       :param gamma:        List of Red, green and blue gamma adjustments, in the list format: (rg, gg, bg).
                            Gamma adjustment—an exponential gain factor. For example, rg=1.2 will brighten the red
                            pixel values and gg=0.8 will darken the green pixel values.
    """
    funcName = 'CMNET2_adjust_rgb'
    rgb = clip
    # unpack rgb_factor
    r = factor[0]
    g = factor[1]
    b = factor[2]
    # unpack rgb_bias
    rb = bias[0]
    gb = bias[1]
    bb = bias[2]
    # unpack rgb_gamma
    rg = gamma[0]
    gg = gamma[1]
    bg = gamma[2]
    if rgb.format.color_family != vs.RGB:
        raise ValueError(funcName + ': input clip needs to be RGB!')

    rgb_type = rgb.format.sample_type
    size = 2 ** rgb.format.bits_per_sample
    # adjusting bias values rb,gb,bb for any RGB bit depth
    limited = is_limited_range(rgb)
    if limited:
        if rb > 235 or rb < -235:
            raise ValueError(funcName + ': source is flagged as "limited" but rb is out of range [-235,235]!')
        if gb > 235 or gb < -235:
            raise ValueError(funcName + ': source is flagged as "limited" but gb is out of range [-235,235]!')
        if bb > 235 or bb < -235:
            raise ValueError(funcName + ': source is flagged as "limited" but bb is out of range [-235,235]!')
    else:
        if rb > 255 or rb < -255:
            raise ValueError(funcName + ': source is flagged as "full" but rb is out of range [-255,255]!')
        if gb > 255 or gb < -255:
            raise ValueError(funcName + ': source is flagged as "full" but gb is out of range [-255,255]!')
        if bb > 255 or bb < -255:
            raise ValueError(funcName + ': source is flagged as "full" but bb is out of range [-255,255]!')

    if rg < 0:
        raise ValueError(funcName + ': rg needs to be >= 0!')
    if gg < 0:
        raise ValueError(funcName + ': gg needs to be >= 0!')
    if bg < 0:
        raise ValueError(funcName + ': bg needs to be >= 0!')

    if limited:
        if rgb_type == vs.INTEGER:
            maxVal = 235
        else:
            maxVal = 235.0
    else:
        if rgb_type == vs.INTEGER:
            maxVal = 255
        else:
            maxVal = 255.0
    rb, gb, bb = map(lambda b: b if size == maxVal else size / maxVal * b if rgb_type == vs.INTEGER else b / maxVal,
                     [rb, gb, bb])

    # x*r + rb , x*g + gb , x*b + bb
    rgb_adjusted = vs.core.std.Expr(rgb, [f"x {r} * {rb} +", f"x {g} * {gb} +", f"x {b} * {bb} +"])
    # gamma per channel
    planes = [vs.core.std.ShufflePlanes(rgb_adjusted, planes=p, colorfamily=vs.GRAY) for p in [0, 1, 2]]
    planes = [vs.core.std.Levels(planes[p], gamma=g) if not g == 1 else planes[p] for p, g in enumerate([rg, gg, bg])]
    rgb_adjusted = vs.core.std.ShufflePlanes(planes, planes=[0, 0, 0], colorfamily=vs.RGB)
    return rgb_adjusted


def rgb_denoise(clip: vs.VideoNode, denoise_levels: list[float] = (0.3, 0.2),
                 rgb_factors: list[float] = (0.98, 1.02, 1.0)) -> vs.VideoNode:
    w_strength = denoise_levels[0]
    b_strength = denoise_levels[1]
    r = rgb_factors[0]
    g = rgb_factors[1]
    b = rgb_factors[2]
    clip = clip.std.Levels(min_in=0, max_in=255, min_out=16, max_out=235)
    clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_in_s="full", range_s="limited")
    # step #1 : rgb colors are normalized and changed using rgb factors (this will change also the contrast/luminosity)
    clip = rgb_balance(clip=clip, strength=w_strength, rgb_factor=[r, g, b])
    clip = clip.std.Levels(min_in=16, max_in=235, min_out=0, max_out=255)
    clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in_s="709", range_in_s="limited", range_s="full")
    return clip


"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description: function that takes a video clip as an input and calculates the average color values 
for each of the three color planes (red, green, blue), with optional RGB scaling factors.
The white balance of the input clip is adjusted based on the color balance of the individual frames.
URL: http://www.vapoursynth.com/doc/functions/frameeval.html
------------------------------------------------------------------------------- 
"""
def rgb_balance(clip: vs.VideoNode, strength: float = 0.5, rgb_factor: list = (1.0, 1.0, 1.0) ) -> vs.VideoNode:
   """ Auto white balance filter using PlaneStats()

    :param clip:           Clip to process (support only RGB24).
    :param strength:       Strength of the filter. A strength=0 means that the clip is returned unchanged,
                           range [0, 1] (default=0.5)
    :param rgb_factor:     List of Red, Green and Blue scaling factor, in the list format: (r, g, b),
                           default = (1, 1, 1). For example, r=1.3 multiplies the red channel pixel values by 1.3.

   """
   rgb_clip = clip

   # A zero weight means that the clip filtered is returned unchanged and 1 means that original clip is returned
   weight: float = min(max(1.0 - strength, 0.0), 1.0)

   # auto white from http://www.vapoursynth.com/doc/functions/frameeval.html
   def frame_autowhite(n, f, clip, core, rgb_fact):
      small_number = 0.000000001
      # unpack rgb_factor
      r = rgb_fact[0]
      g = rgb_fact[1]
      b = rgb_fact[2]
      red = f[0].props['PlaneStatsAverage']
      green = f[1].props['PlaneStatsAverage']
      blue = f[2].props['PlaneStatsAverage']
      max_rgb = max(red, green, blue)
      red_corr = max_rgb / max(red, small_number)
      green_corr = max_rgb / max(green, small_number)
      blue_corr = max_rgb / max(blue, small_number)
      norm = max(blue, math.sqrt(red_corr * red_corr + green_corr * green_corr + blue_corr * blue_corr) / math.sqrt(3),
                 small_number)
      r_gain = round(r * red_corr / norm, 8)
      g_gain = round(g * green_corr / norm, 8)
      b_gain = round(b * blue_corr / norm, 8)
      return core.std.Expr(clip,
                           expr=['x ' + repr(r_gain) + ' *', 'x ' + repr(g_gain) + ' *', 'x ' + repr(b_gain) + ' *'])

   r_avg = vs.core.std.PlaneStats(rgb_clip, plane=0)
   g_avg = vs.core.std.PlaneStats(rgb_clip, plane=1)
   b_avg = vs.core.std.PlaneStats(rgb_clip, plane=2)

   clip_a = vs.core.std.FrameEval(rgb_clip, partial(frame_autowhite, clip=rgb_clip, core=vs.core,
                                                           rgb_fact=rgb_factor), prop_src=[r_avg, g_avg, b_avg])

   clip_b = rgb_clip

   # A zero weight means that clip_a is returned unchanged and 1 means that clip_b is returned unchanged
   if 0 <= weight < 1:
      clip_rgb = vs.core.std.Merge(clip_a, clip_b, weight)
   else:
      clip_rgb = rgb_clip  # is returned the original clip

   if clip.format.id != vs.RGB24:
      # convert the format for tweak to YUV 8bits
      clip_new = clip_rgb.resize.Bicubic(format=vs.YUV420P8, matrix_s="709", range_s="limited")
   else:
      clip_new = clip_rgb

   return clip_new


"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2026-06-20
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Standalone PIL-to-PIL colorization using CMNET2.
Takes a color reference image and a B&W target image, returns the colorized target.
"""


def pil_cmnet2_colorize(
    ref_image: Image.Image,
    bw_image: Image.Image,
    image_size: int = -1,
    project_dir: str = None,
) -> Image.Image:
    """Colorize a B&W target image using a color reference image via CMNET2.

    This function provides a simple PIL-to-PIL interface for exemplar-based
    video colorization. The reference image supplies color context; the model
    propagates those colors to the target B&W frame using the CMNET2 deep
    learning model with DINOv2 features and permanent-memory attention.

    :param ref_image:   PIL RGB Image -- the color reference frame.
                        Must be mode 'RGB'. If 'L' (grayscale) is passed,
                        it is converted to 'RGB' automatically.
    :param bw_image:    PIL Image -- the B&W target frame to colorize.
                        Must be mode 'RGB' (L channel replicated into 3 channels)
                        or 'L' (grayscale, auto-converted to 'RGB').
    :param image_size:  Inference resolution override. -1 uses the original
                        image dimensions. Larger values increase quality at
                        the cost of VRAM and speed.
    :param project_dir: Path to the vscmnet2 package root directory.
                        When None (default), auto-detects from this file's
                        location. Only override when the package is relocated
                        at runtime.

    :return:            Colorized PIL RGB Image at the same resolution as
                        the input bw_image.

    Notes:
        - The first call loads the CMNET2 model from disk (~5-10 s).
          Subsequent calls in the same process reuse the loaded model and
          are much faster (~1-2 s per image).
        - Requires a CUDA-capable GPU with PyTorch >= 2.9.1.
        - Model weights must be present under vscmnet2/weights/ and
          vscmnet2/models/checkpoints/ (see README).

    Example::

        from PIL import Image
        from vscmnet2 import pil_cmnet2_colorize

        ref = Image.open("reference_frame.png").convert("RGB")
        bw  = Image.open("target_frame.png").convert("RGB")
        result = pil_cmnet2_colorize(ref, bw)
        result.save("colorized.png")
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("pil_cmnet2_colorize: CUDA is not available")

    # Ensure consistent CUDA environment (idempotent).
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    # Normalize image modes.
    if ref_image.mode != "RGB":
        ref_image = ref_image.convert("RGB")
    if bw_image.mode != "RGB":
        bw_image = bw_image.convert("RGB")

    # Resolve project directory for model weights.
    if project_dir is None:
        project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    # Lazy import to avoid circular dependencies at module level.
    from .colormnet2.colormnet2_render import ColorMNetRender2

    # Reset any existing singleton so we get a clean instance with our params.
    # This ensures correct image_size and single-frame vid_length even when a
    # VapourSynth session previously initialized the singleton differently.
    ColorMNetRender2.reset()

    # Create the renderer tuned for single-image colorization.
    render = ColorMNetRender2(
        image_size=image_size,
        vid_length=1,
        enable_resize=False,
        encode_mode=1,           # local / in-process
        max_memory_frames=1,
        reset_on_ref_update=False,
        retry_perm_share_threshold=-1.0,  # disable retry
        project_dir=project_dir,
    )

    # Set the color reference and colorize the target.
    render.set_ref_frame(ref_image, frame_propagate=False)
    result = render.colorize_frame(ti=0, frame_i=bw_image)

    # Clean up GPU memory but keep the singleton alive for reuse.
    render.reset_state()

    return result