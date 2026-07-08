"""Runtime checks for optional local LLM backends."""

from __future__ import annotations

import importlib.util
import re


def _version_tuple(version: str) -> tuple[int, ...]:
    match = re.match(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", version)
    if not match:
        return (0,)
    return tuple(int(part) for part in match.groups(default="0"))


def llm_backend_error(*, min_torch_version: str = "2.4") -> str | None:
    """Return a user-readable reason if Transformers cannot use PyTorch."""

    if importlib.util.find_spec("torch") is None:
        return "PyTorch is not installed. Install the LLM extras before using --enable-llm."
    try:
        import torch
    except Exception as exc:
        return f"PyTorch could not be imported: {exc}"

    torch_version = str(getattr(torch, "__version__", "0"))
    if _version_tuple(torch_version) < _version_tuple(min_torch_version):
        return (
            f"PyTorch {torch_version} is installed, but Transformers requires PyTorch >= {min_torch_version} "
            "for this local LLM path."
        )
    return None
