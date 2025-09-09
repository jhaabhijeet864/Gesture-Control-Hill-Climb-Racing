import json
import os
import logging
from pynput.keyboard import Key


log = logging.getLogger("gesture-app")


DEFAULT_CONFIG = {
    "min_detection_confidence": 0.7,
    "min_tracking_confidence": 0.7,
    "max_num_hands": 2,
    "smoothing": 8,
    "pinch_threshold": 0.05,
    "ready_threshold": 0.10,
    "click_cooldown_frames": 5,
    "keys": {
        "gas": "right",
        "brake": "left"
    }
}


CONFIG_PATH = "config.json"


def save_config(cfg, path: str = CONFIG_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def load_config(path: str = CONFIG_PATH):
    if not os.path.exists(path):
        save_config(DEFAULT_CONFIG, path)
        return DEFAULT_CONFIG.copy()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Fill any missing defaults
    merged = DEFAULT_CONFIG.copy()
    for k, v in data.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            merged[k].update(v)
        else:
            merged[k] = v
    return merged


def key_from_name(name: str):
    name = (name or "").lower()
    try:
        return getattr(Key, name)
    except AttributeError:
        if log:
            log.warning("Unknown key in config: %s, defaulting to 'right'", name)
        return Key.right
