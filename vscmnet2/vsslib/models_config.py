"""
-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
Model file names used by the filter, loaded from the data file models.json
(stored next to this module).

The checkpoint file names are part of the contract with the model releases
(see the "Models Download" section of the README): keeping them in a data file
instead of hardcoding them in the code gives a single point of truth, easy to
inspect and to update.  models.json overrides the DEFAULTS defined below.

models.json is optional: if it is missing, unreadable or malformed, DEFAULTS
is used and (best effort) a warning is queued in the CMNET2 log buffer.
-------------------------------------------------------------------------------
"""
import json
from pathlib import Path

from ..colormnet2.colormnet2_logbuffer import log_warning as _buf_warning

# Fallback values (same as models.json), used when models.json is missing or
# does not define the requested entry.  Keep in sync with models.json.
DEFAULTS = {
    "cmnet2": {
        "dinov3": {
            "checkpoint": "DINOv3FeatureV6_LocalAtten_p374099.pth",
            "weights_dir": "dinov3-vitb16",
            "enable_proximity_bias": False,
            "proximity_bias_alpha": 0.5,
        },
        "dinov2": {
            "checkpoint": "DINOv2FeatureV6_LocalAtten_s2_154000.pth",
        },
    },
}

_CONFIG_PATH = Path(__file__).with_name("models.json")


def _warn(msg: str) -> None:
    """Queue a models_config warning into the shared CMNET2 log buffer.

    Python logging is deliberately not used here: vsscript hosts (vspipe,
    vsViewer) cannot route logging from a non-main thread - in remote mode
    the model build runs inside the RPC server thread - and the VS logging
    bridge would raise AttributeError ('PythonVSScriptLoggingBridge' object
    has no attribute 'parent'), aborting the server-side initialize. Same
    pattern as model/network.py::_warn (load_weights remap warning).
    """
    _buf_warning(msg)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load() -> dict:
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("the root of models.json must be an object")
        return _deep_merge(DEFAULTS, data)
    except (OSError, ValueError) as e:
        _warn(f"models_config: cannot read '{_CONFIG_PATH}' ({type(e).__name__}: {e}), "
              f"using the built-in defaults")
        return _deep_merge(DEFAULTS, {})


MODELS = _load()


def get_cmnet2_model(backbone: str) -> dict:
    """Return the model entry for a CMNET2 backbone ('dinov2' or 'dinov3').

    The returned dict always contains 'checkpoint' (file name inside the
    weights directory) and may contain 'weights_dir' (directory holding the
    auxiliary backbone files, e.g. dinov3-vitb16).
    """
    try:
        entry = MODELS["cmnet2"][backbone]
    except KeyError:
        raise ValueError(f"models_config: no CMNET2 entry for backbone {backbone!r}")
    if not isinstance(entry, dict) or not entry.get("checkpoint"):
        raise ValueError(f"models_config: invalid CMNET2 entry for backbone {backbone!r}")
    return entry


def check_file(path: str, what: str = "model file") -> str:
    """Return path unchanged when it exists, otherwise raise a clear error.

    On failure the error also lists the files actually present in the
    directory, so a wrong/misspelled checkpoint name is immediately visible.
    """
    p = Path(path)
    if not p.is_file():
        listing = ""
        try:
            names = sorted(f.name for f in p.parent.iterdir() if f.is_file())
            listing = ("\n  files present in " + str(p.parent) + ": " + ", ".join(names)) \
                if names else ("\n  (no files in " + str(p.parent) + ")")
        except OSError:
            pass
        raise FileNotFoundError(f"{what} not found: {path}{listing}")
    return path
