"""
-------------------------------------------------------------------------------
Author: Dan64
Date: 2025-10-04
version:
LastEditors: Dan64
LastEditTime: 2026-05-01
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
CMNET2 functions library.
"""
import os

# Hybrid paths for plugins used by CMNET2
vsslib_dir: str = os.path.dirname(os.path.realpath(__file__))

support_dir: str = os.path.join(vsslib_dir, "..", "plugins", "Support")

# Path for SCDetect:
MiscFilter_dir: str = os.path.join(vsslib_dir, "..", "plugins", "MiscFilter", "MiscFilters")

# Path for LSMASHSource:
LSMASHSource_dir: str = os.path.join(vsslib_dir, "..", "plugins", "SourceFilter", "LSmashSource")
