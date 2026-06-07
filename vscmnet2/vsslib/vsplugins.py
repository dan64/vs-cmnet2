"""
------------------------------------------------------------------------------- 
Author: Dan64
Date: 2024-10-09
version: 
LastEditors: Dan64
LastEditTime: 2026-06-07
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Utility functions to load Vapoursynth plugins dynamically.
"""

import vapoursynth as vs
from pathlib import Path
from .vsutils import CMNET2_LogMessage, MessageType, frame_to_image

from .constants import *

from .__int__ import *

"""
------------------------------------------------------------------------------- 
Author: Dan64
------------------------------------------------------------------------------- 
Description:
------------------------------------------------------------------------------- 
Utility functions to load Vapoursynth plugins dynamically.
"""
def load_TCanny_plugin() -> bool:
    """
    Ensures TCanny VapourSynth plugin is loaded.
    URL: https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TCanny
    """

    plugin_path = os.path.normpath(os.path.join(support_dir, "TCanny.dll"))

    try:
        if hasattr(vs.core, 'tcanny') and hasattr(vs.core.akarin, 'TCanny'):
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                CMNET2_LogMessage(MessageType.INFORMATION,f"[INFO] Plugin 'TCanny' already loaded.")
            return True
        else:
            vs.core.std.LoadPlugin(path=plugin_path)
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                CMNET2_LogMessage(MessageType.INFORMATION, f"[INFO] Plugin 'TCanny' loaded from: {plugin_path}")
            return True
    except Exception as error:
        CMNET2_LogMessage(MessageType.WARNING,"[WARNING] Plugin 'TCanny': check/load failed ->", str(error))
        return False

def load_Akarin_plugin() -> bool:
    """
    Ensures Akarin VapourSynth plugin is loaded.
    URL: https://github.com/AkarinVS/vapoursynth-plugin
    """

    plugin_path = os.path.normpath(os.path.join(support_dir, "akarin.dll"))

    try:
        if hasattr(vs.core, 'akarin') and hasattr(vs.core.akarin, 'Expr'):
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                CMNET2_LogMessage(MessageType.INFORMATION,f"[INFO] Plugin 'Akarin' already loaded.")
            return True
        else:
            vs.core.std.LoadPlugin(path=plugin_path)
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                CMNET2_LogMessage(MessageType.INFORMATION, f"[INFO] Plugin 'Akarin' loaded from: {plugin_path}")
            return True
    except Exception as error:
        CMNET2_LogMessage(MessageType.WARNING,"[WARNING] Plugin 'Akarin': check/load failed ->", str(error))
        return False


def load_SCDetect_plugin() -> bool:
    """
    Ensures SCDetect VapourSynth plugin is loaded.
    URL: https://github.com/vapoursynth/vs-miscfilters-obsolete
    """

    plugin_path = os.path.normpath(os.path.join(MiscFilter_dir, "MiscFilters.dll"))

    try:
        if hasattr(vs.core, 'misc') and hasattr(vs.core.misc, 'SCDetect'):
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                CMNET2_LogMessage(MessageType.INFORMATION,"[INFO] Plugin 'SCDetect' already loaded.")
            return True
        else:
            vs.core.std.LoadPlugin(path=plugin_path)
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                CMNET2_LogMessage(MessageType.INFORMATION, f"[INFO] Plugin 'SCDetect' loaded from: {plugin_path}")
            return True
    except Exception as error:
        CMNET2_LogMessage(MessageType.INFORMATION, "[WARNING] Plugin 'SCDetect': check/load failed ->", str(error))
        return False

def load_LSMASHSource_plugin() -> bool:
    """
    Ensures LSMASHSource VapourSynth plugin is loaded.
    URL: https://github.com/AkarinVS/L-SMASH-Works
    """

    plugin_path = os.path.normpath(os.path.join(LSMASHSource_dir, "LSMASHSource.dll"))

    try:
        if hasattr(vs.core, 'lsmas') and hasattr(vs.core.lsmas, 'LWLibavSource'):
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                CMNET2_LogMessage(MessageType.INFORMATION,"[INFO] Plugin 'LSMASHSource' already loaded.")
            return True
        else:
            vs.core.std.LoadPlugin(path=plugin_path)
            if DEF_DEBUG_LEVEL > DEF_LEVEL_NONE:
                CMNET2_LogMessage(MessageType.INFORMATION, f"[INFO] Plugin 'LSMASHSource' loaded from: {plugin_path}")
            return True
    except Exception as error:
        CMNET2_LogMessage(MessageType.WARNING,"[WARNING] Plugin 'LSMASHSource': check/load failed ->", str(error))
        return False
