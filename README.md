# vs-cmnet2

**VapourSynth filter for exemplar-based video colorization using CMNET2.**

Colorizes black-and-white clips by propagating color from reference frames using the [CMNET2](https://github.com/dan64/cmnet2) deep learning model with a sliding permanent-memory window.

---

## Installation

Download the latest wheel from [Releases](https://github.com/dan64/vs-cmnet2/releases) and install:

```bash
pip install vscmnet2-1.0.0-py3-none-any.whl
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
- **CUDA-capable GPU** with PyTorch ≥ 2.9.1

---

## Model Weights

Download the following files from the [CMNET2 v1.0.0 Release](https://github.com/dan64/cmnet2/releases/tag/v1.0.0) and place them in the correct directories under `vscmnet2/`:

| File | Destination | Download |
|---|---|---|
| `DINOv2FeatureV6_LocalAtten_s2_154000.pth` | `vscmnet2/weights/` | [download](https://github.com/dan64/cmnet2/releases/download/v1.0.0/DINOv2FeatureV6_LocalAtten_s2_154000.pth) |
| `dinov2_vits14_pretrain.pth` | `vscmnet2/models/checkpoints/` | [download](https://github.com/dan64/cmnet2/releases/download/v1.0.0/dinov2_vits14_pretrain.pth) |
| `resnet18-5c106cde.pth` | `vscmnet2/models/checkpoints/` | [download](https://github.com/dan64/cmnet2/releases/download/v1.0.0/resnet18-5c106cde.pth) |
| `resnet50-19c8e357.pth` | `vscmnet2/models/checkpoints/` | [download](https://github.com/dan64/cmnet2/releases/download/v1.0.0/resnet50-19c8e357.pth) |

> **Note:** The DINOv2 source code (`facebookresearch_dinov2_main/`) is already included in this repository under `vscmnet2/models/`.

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

### Read external video

```python
from vscmnet2 import vs_read_video

clip = vs_read_video("/path/to/video.mkv")
```

---

## Key Parameters

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
| `torch_dir` | str | model dir | Torch hub cache location |

### `vs_cmnet2dit`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `clip` | VideoNode | — | B&W input clip |
| `sc_thresh` | float | `0.035` | Scene-detect threshold |
| `sc_min_int` | int | `25` | Min frame distance between scene changes |
| `max_memory_frames` | int | `0` (→20) | Permanent-memory window (even, pair-wise) |
| `dit_engine_params` | dict | `None` | DiT Engine Server connection |

---

## Model Architecture

CMNET2 (Colorization Memory Network v2) is an exemplar-based video colorization model. It maintains a **sliding permanent memory** of reference frames and propagates color through a space-time memory network. The architecture uses:

- **DINOv2 ViT-S/14** as the key encoder backbone
- **ResNet-18** and **ResNet-50** as value encoders
- **LocalGatedPropagation** for attention-based memory readout
- **CBAM** (Convolutional Block Attention Module) for feature refinement
- **KeyValueMemoryStore** with top-k readout for efficient retrieval

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
- **DINOv2**: [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- **XMem**: [hkchengrex/XMem](https://github.com/hkchengrex/XMem) — Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model

---

## License

MIT
