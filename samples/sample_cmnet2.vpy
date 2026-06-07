# Imports
import vapoursynth as vs
# getting Vapoursynth core
import sys
import os
core = vs.core
# Limit frame cache to 16384MB
core.max_cache_size = 16384
# Set Paths
clip_dir = os.path.dirname(os.path.realpath(__file__))
VideoPath=os.path.join(clip_dir, "sample_bw.mp4")
RefDir=os.path.join(clip_dir, "references")
# Import scripts
import vscmnet2 as cmnet2
clip = cmnet2.vs_read_video(source=VideoPath)
# adding colors using HAVC
clip = cmnet2.vs_cmnet2(clip, method=4, sc_framedir=RefDir, encode_mode=0, max_memory_frames=10, ref_mode=1)
# adjusting output color from RGB24 to YUV420P10 for x265Model 
clip = core.resize.Bicubic(clip=clip, format=vs.YUV420P10, matrix_s="709", range_in_s="full", range_s="limited", dither_type="error_diffusion") 
# output
clip.set_output()


