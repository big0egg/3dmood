"""Lazy adapter loading: offline evaluation never imports torch/3D-MOOD."""

import importlib

from ..common import file_digest


def create_adapter(name, options):
    target = "ioutest.adapters.mood3d:Mood3DAdapter" if name == "3dmood" else name
    if ":" not in target:
        raise ValueError("adapter 使用 3dmood 或 your_module:YourAdapter")
    module_name, attribute = target.rsplit(":", 1)
    module = importlib.import_module(module_name)
    adapter = getattr(module, attribute)(options)
    for method in ("load", "predict", "close"):
        if not callable(getattr(adapter, method, None)):
            raise ValueError(f"模型适配器缺少 {method} 方法")
    return adapter, {"target": target, "adapter_file_sha256": file_digest(module.__file__)}
