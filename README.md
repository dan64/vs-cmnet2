# vs-cmnet2

**VapourSynth filter for exemplar-based video colorization using CMNET2.**

Colorizes black-and-white clips by propagating color from reference frames using the [CMNET2](https://github.com/dan64/cmnet2) deep learning model with a sliding permanent-memory window.

---

## Installation

Download the latest wheel from [Releases](https://github.com/dan64/vs-cmnet2/releases) and install:

```bash
pip install vscmnet2-1.0.9-py3-none-any.whl
```

### Plugins setup

Download `plugins_win.zip` from the [Release v1.0.0](https://github.com/dan64/vs-cmnet2/releases/download/v1.0.0/plugins_win.zip) and extract it into `vscmnet2/plugins/`. The resulting tree will be:

```
vscmnet2/plugins/
├── Support/
│   ├── TCanny.dll          # Edge detection
│   └── akarin.dll          # Expression evaluation
├── MiscFilter/MiscFilters/
│   └── MiscFilters.dll     # Scene-change detection (SCDetect)
└── SourceFilter/LSmashSource/
    ├── LSMASHSource.dll     # Video file reader
    ├── vcruntime140.dll
    └── vcruntime140_1.dll
```

### Model weights

See [Model Weights](#model-weights) below.

---

## Requirements

- **Python** ≥ 3.12
- **VapourSynth** ≥ R74
- **CUDA-capable GPU** with PyTorch ≥ 2.7.0 (fully tested with 2.7.1+cu128 and 2.10+cu130)
- **safetensors** ≥ 0.4 (used by the native DINOv3 backbone loader)

---

## Model Weights

### DINOv3 backbone (default)

The filter uses the **DINOv3 ViT-B/16** key-encoder backbone by default (new in this release). Download these files from the [CMNET2 v1.1.0 Release](https://github.com/dan64/cmnet2/releases/tag/v1.1.0):

| File | Destination | Download |
|---|---|---|
| `DINOv3FeatureV6_LocalAtten_p374099.pth` | `vscmnet2/weights/` | [download](https://github.com/dan64/cmnet2/releases/download/v1.2.0/DINOv3FeatureV6_LocalAtten_p374099.pth) |
| `dinov3-vitb16.zip` (extract to `vscmnet2/weights/`) | `vscmnet2/weights/dinov3-vitb16/` | [download](https://github.com/dan64/cmnet2/releases/download/v1.1.0/dinov3-vitb16.zip) |

### DINOv2 backbone (legacy)

To use the previous **DINOv2 ViT-S/14** backbone, pass `backbone="dinov2"` to any of the filter functions and download these files from the [CMNET2 v1.0.0 Release](https://github.com/dan64/cmnet2/releases/tag/v1.0.0):

| File | Destination | Download |
|---|---|---|
| `DINOv2FeatureV6_LocalAtten_s2_154000.pth` | `vscmnet2/weights/` | [download](https://github.com/dan64/cmnet2/releases/download/v1.0.0/DINOv2FeatureV6_LocalAtten_s2_154000.pth) |
| `dinov2_vits14_pretrain.pth` | `vscmnet2/models/checkpoints/` | [download](https://github.com/dan64/cmnet2/releases/download/v1.0.0/dinov2_vits14_pretrain.pth) |
| `resnet18-5c106cde.pth` | `vscmnet2/models/checkpoints/` | [download](https://github.com/dan64/cmnet2/releases/download/v1.0.0/resnet18-5c106cde.pth) |
| `resnet50-19c8e357.pth` | `vscmnet2/models/checkpoints/` | [download](https://github.com/dan64/cmnet2/releases/download/v1.0.0/resnet50-19c8e357.pth) |

> **Note:** The DINOv2 source code (`facebookresearch_dinov2_main/`) is already included in this repository under `vscmnet2/models/`. The DINOv3 backbone is loaded by a native PyTorch implementation (`colormnet2/model/dinov3_vit.py`) from the local `vscmnet2/weights/dinov3-vitb16/` directory (self-contained, no `transformers` dependency, never from the global HuggingFace cache).

### Model file names (`models.json`)

The names of the checkpoints are not hardcoded in the code: they are stored in a single data file, `vscmnet2/vsslib/models.json`, shipped with the package:

```json
{
  "cmnet2": {
    "dinov3": {
      "checkpoint": "DINOv3FeatureV6_LocalAtten_p374099.pth",
      "weights_dir": "dinov3-vitb16",
      "enable_proximity_bias": false,
      "proximity_bias_alpha": 0.5
    },
    "dinov2": {
      "checkpoint": "DINOv2FeatureV6_LocalAtten_s2_154000.pth"
    }
  }
}
```

Normally there is no need to touch it. Edit it only if the checkpoint files have different names (custom or renamed weights) or you want to change the default proximity-bias settings: `checkpoint` is the file inside `vscmnet2/weights/`, `weights_dir` is the auxiliary directory used by the DINOv3 backbone, and `enable_proximity_bias`/`proximity_bias_alpha` set the default for [proximity-weighted memory matching](#proximity-weighted-memory-matching-optional-dinov3-only) (DINOv3 only — ignored for DINOv2). When the configured file is missing, initialization stops immediately and the error lists the files actually present in the weights directory — so a typo in the checkpoint name (e.g. `LocalAttn` instead of `LocalAtten`) is immediately visible instead of failing silently.

If `models.json` is missing or malformed, the built-in default names (the ones listed above) are used, and a warning is queued in the CMNET2 log buffer (the same channel used for the other model-build warnings, forwarded to the VapourSynth log both locally and in remote/`encode_mode=0`).

### Proximity-weighted memory matching (optional, DINOv3 only)

By default, permanent-memory candidates are ranked purely by content similarity, with no notion of *when* in the video a reference frame was captured relative to the frame being colorized — with a wide `max_memory_frames` window holding several visually similar but differently-colored references, this can wash the result toward gray. From CMNET2 v1.1.0, the model can optionally favor temporally closer references instead, without ever reducing the permanent memory's overall contribution to the readout. See the [mechanism explanation and a visual example](https://github.com/dan64/cmnet2#proximity-weighted-memory-matching-optional-dinov3-only) in the CMNET2 README.

Like `backbone`, this can be controlled per-call via the `enable_proximity_bias`/`proximity_bias_alpha` parameters of `vs_cmnet2` (both default to `None`, meaning "use whatever `vsslib/models.json` says"). It is **not** exposed on `vs_cmnet2_recolor`/`vs_cmnet2dit` — those only pick up the installation-wide default from `vsslib/models.json` (see above). Off by default. To change the installation-wide default (useful for permanent memory window size > 50) it is necessary to set `enable_proximity_bias=true` in the configuration 
file stored in: `vsslib/models.json` as shown in the example below:

```json
{
  "cmnet2": {
    "dinov3": {
      "checkpoint": "DINOv3FeatureV6_LocalAtten_p374099.pth",
      "weights_dir": "dinov3-vitb16",
      "enable_proximity_bias": true,
      "proximity_bias_alpha": 0.5
    },
    "dinov2": {
      "checkpoint": "DINOv2FeatureV6_LocalAtten_s2_154000.pth"
    }
  }
}
```

To enable it only for a specific `vs_cmnet2` call instead of installation-wide, pass the parameters directly (they take precedence over `vsslib/models.json`):

```python
clip = vs_cmnet2(
    clip,
    clip_ref=ref_clip,
    method=0,
    enable_proximity_bias=True,
    proximity_bias_alpha=0.5,
)
```


## Install spatial_correlation_sampler

In the [Release 1.0.0](https://github.com/dan64/vs-cmnet2/releases/download/v1.0.0/spatial_correlation_sampler-0.5.0-cp312-cp312-win_amd64_torch-2.10+cu130.zip) there is an archive with a compiled version (PyTorch 2.10 + CUDA 13.0) of [Pytorch-Correlation-extension](https://github.com/ClementPinard/Pytorch-Correlation-extension), required by `vscmnet2` for temporal alignment during encoding:

```powershell
pip install spatial_correlation_sampler-0.5.0-cp312-cp312-win_amd64.whl
```

> The wheel is pre-built for **Python 3.12 / PyTorch 2.10+cu130 / Windows x64**.
> It will only work with that exact combination. For other environments it will be necessary build the wheel from sources
> (e.g. for Hybrid R77 with PyTorch 2.7.1+cu128, build it against the matching PyTorch version).

### 4. DiT model (optional — for `vs_cmnet2dit`)

The DiT path uses a **DiT Engine Server** running separately. Start the server pointing to a [Nunchaku](https://github.com/mit-han-lab/nunchaku) SVD quant model, then connect via:

```python
clip = vs_cmnet2dit(clip, dit_engine_params={"host": "127.0.0.1", "port": 8765})
```

---

## Usage

### Basic colorization with external reference clip

```python
from vscmnet2 import vs_cmnet2
clip = vs_cmnet2(clip, clip_ref=ref_clip, method=6)
```

### Reference frames from a directory

Reference frames are read from a folder. Files must be named `ref_NNNNNN.png` (e.g. `ref_000897.png`).

```python
clip = vs_cmnet2(clip, sc_framedir="/path/to/refs", method=4)
```

### Custom render speed and retry

```python
clip = vs_cmnet2(
    clip,
    clip_ref=ref_clip,
    method=0,
    render_speed="Slow",
    render_vivid=True,
    max_memory_frames=40,
    retry_threshold=0.35,
    retry_model=1,            # Dit model
    backbone="dinov3",        # default; use "dinov2" for the legacy backbone
)
```

### DiT-based colorization

```python
from vscmnet2 import vs_cmnet2dit

clip = vs_cmnet2dit(
    clip,
    dit_engine_params={
        "host": "127.0.0.1",
        "port": 8765,
    },
    max_memory_frames=20,
)
```

### Re-color a range of frames

Re-colorizes only the frames between two reference frames, leaving the rest unchanged. Useful for fixing specific sections of an already colored clip.

```python
from vscmnet2 import vs_cmnet2_recolor

clip = vs_cmnet2_recolor(
    clip,
    ref_framedir="/path/to/refs",
    ref_start_path="/path/to/refs/ref_000100.png",
    ref_end_path="/path/to/refs/ref_000200.png",
    method=4,
    max_memory_frames=20,
    backbone="dinov3",        # default; use "dinov2" for the legacy backbone
)
```

### Read external video

```python
from vscmnet2 import vs_read_video

clip = vs_read_video("/path/to/video.mkv")
```

---

## Key Parameters

### `vs_cmnet2_recolor`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `clip` | VideoNode | — | Already colorized clip to re-color |
| `method` | int | `4` | 3=ref same as video, 4=ref different from video |
| `render_speed` | str | `"auto"` | `auto`, `fast`, `medium`, `slow`, `slower` |
| `render_vivid` | bool | `False` | +15% saturation boost |
| `ref_framedir` | str | — | Directory with reference frames (format: ref_NNNNNN.png) |
| `ref_start_path` | str | — | First reference frame to re-color from |
| `ref_end_path` | str | — | Last reference frame to re-color to |
| `max_memory_frames` | int | `0` (→20) | Permanent-memory window size (even, 10–500) |
| `retry_threshold` | float | `0.0` | Retry trigger (0.0=disabled; suggest 0.20–0.35) |
| `retry_model` | int | `1` | 1=DiT fp4, 2=DiT int4 |
| `backbone` | str | `"dinov3"` | Key-encoder backbone: `dinov3` (default) or `dinov2` |
| `torch_dir` | str | model dir | Torch hub cache location |

### `vs_cmnet2`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `clip` | VideoNode | — | B&W input clip |
| `clip_ref` | VideoNode | `None` | Reference clip (method 5,6) |
| `method` | int | `0` | Reference frame generation: 3-4=external, 5-6=clipRef |
| `render_speed` | str | `"auto"` | `auto`, `fast`, `medium`, `slow`, `slower` |
| `render_vivid` | bool | `False` | +15% saturation boost |
| `encode_mode` | int | `0` | 0=remote (recommended), 1=local |
| `max_memory_frames` | int | `0` (→20) | Permanent-memory window size (even, 10–500) |
| `ref_mode` | int | `1` | 0=direct folder, 1=VS clips |
| `retry_threshold` | float | `0.0` | Retry trigger (0.0=disabled; suggest 0.20–0.35) |
| `retry_model` | int | `0` | 0=DeOldify+DDColor, 1=DiT fp4, 2=DiT int4 |
| `backbone` | str | `"dinov3"` | Key-encoder backbone: `dinov3` (default) or `dinov2` |
| `enable_proximity_bias` | bool | `None` | Enable [proximity-weighted memory matching](#proximity-weighted-memory-matching-optional-dinov3-only) (DINOv3 only). `None` = use `vsslib/models.json` |
| `proximity_bias_alpha` | float | `None` | Strength of the proximity bias. `None` = use `vsslib/models.json` |
| `torch_dir` | str | model dir | Torch hub cache location |

### `vs_cmnet2dit`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `clip` | VideoNode | — | B&W input clip |
| `sc_thresh` | float | `0.035` | Scene-detect threshold |
| `sc_min_int` | int | `25` | Min frame distance between scene changes |
| `max_memory_frames` | int | `0` (→20) | Permanent-memory window (even, pair-wise) |
| `dit_engine_params` | dict | `None` | DiT Engine Server connection |
| `backbone` | str | `"dinov3"` | Key-encoder backbone: `dinov3` (default) or `dinov2` |

---

## Model Architecture

CMNET2 (Colorization Memory Network v2) is an exemplar-based video colorization model. It maintains a **sliding permanent memory** of reference frames and propagates color through a space-time memory network. The architecture uses:

- **DINOv3 ViT-B/16** as the key encoder backbone (default, fully fine-tuned) or **DINOv2 ViT-S/14** (legacy, selected with `backbone="dinov2"`)
- **ResNet-18** and **ResNet-50** as value encoders
- **LocalGatedPropagation** for attention-based memory readout
- **CBAM** (Convolutional Block Attention Module) for feature refinement
- **KeyValueMemoryStore** with top-k readout for efficient retrieval, optionally weighted by temporal proximity to reference frames (see [Proximity-weighted memory matching](#proximity-weighted-memory-matching-optional-dinov3-only))

The DiT variant offloads reference-frame colorization to an external DiT (Diffusion Transformer) model running in a separate RPC server process.

---

## Project Structure

```
vscmnet2/
├── __init__.py          # Main VapourSynth wrapper (vs_cmnet2, vs_cmnet2dit, vs_merge, vs_read_video)
├── cmnet2_utils.py      # Format conversion, luma protection, video I/O
├── colormnet2/          # CMNET2 core (colorization engine)
│   ├── __init__.py      # vs_colormnet2_local / vs_colormnet2_remote
│   ├── colormnet2_render.py   # Render class (ColorMNetRender2)
│   ├── colormnet2_server.py   # XML-RPC server
│   ├── colormnet2_client.py   # XML-RPC client
│   ├── model/           # Neural network modules
│   │   ├── network.py   # ColorMNet (top-level nn.Module)
│   │   ├── resnet.py    # ResNet backbone with DINOv2 key encoder
│   │   ├── modules.py   # Key/value encoders, decoder, memory read
│   │   ├── attention.py # LocalGatedPropagation
│   │   └── ...
│   └── inference/       # Inference core, memory manager
├── vsslib/              # Shared VapourSynth utility library
│   ├── models.json      # Checkpoint file names (see "Model file names")
│   ├── models_config.py # models.json loader (get_cmnet2_model, check_file)
│   ├── vsmodels.py      # Model dispatchers (vs_colormnet2, vs_colormnet2dit)
│   ├── vsimage_engine.py   # DiT engine / DeOldify+DDColor fallback
│   ├── vsplugins.py     # VapourSynth plugin loaders
│   ├── vsfilters.py     # VapourSynth filter functions (merge, tweak, etc.)
│   ├── vsscdect.py      # Scene-change detection
│   ├── vsscdetect_edge.py  # Edge-based scene detection
│   └── ...
├── weights/             # CMNET2 model weights
├── models/
│   ├── checkpoints/     # Backbone weights (DINOv2, ResNet)
│   └── facebookresearch_dinov2_main/  # DINOv2 source
└── plugins/             # VapourSynth .dll plugins (from plugins_win.zip)
```

---

## Credits

- **CMNET2**: [dan64/cmnet2](https://github.com/dan64/cmnet2) — Exemplar-based Video Colorization with Long-term Spatiotemporal Memory
- **DINOv3**: [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
- **DINOv2**: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- **XMem**: [hkchengrex/XMem](https://github.com/hkchengrex/XMem) — Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model

---

## License

MIT
