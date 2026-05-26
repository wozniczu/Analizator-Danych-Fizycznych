##
## @file chart_output_names.py
## @brief Mapowanie sztucznych nazw plików z kontenera (np. cfile_...) na nazwy z plt.savefig w kodzie.

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

_IMG_EXT = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp"})
_IMAGE_SIGNATURES = (
    b"\x89PNG",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
)

_SAVEFIG_LITERAL_RE = re.compile(
    r"\.savefig\s*\(\s*r?(['\"])(?P<p>[^'\"]+)\1(?:\s*,|\s*\))",
    re.MULTILINE,
)


def _is_image_entry(filename: str, data: bytes) -> bool:
    if filename and filename.lower().endswith(tuple(_IMG_EXT)):
        return True
    if len(data) >= 4:
        return any(data.startswith(s) for s in _IMAGE_SIGNATURES)
    return False


def is_container_style_filename(filename: str) -> bool:
    """!
    @brief True gdy nazwa wygląda na identyfikator kontenera API, a nie na nazwę z savefig.
    """
    if not filename or not isinstance(filename, str):
        return True
    base = os.path.basename(filename.strip())
    low = base.lower()
    if low.startswith("cfile_"):
        return True
    if low.startswith("ci_image_"):
        return True
    stem, ext = os.path.splitext(low)
    if ext in _IMG_EXT and re.fullmatch(r"[a-f0-9]{20,}", stem):
        return True
    return False


def _sanitize_download_basename(path: str) -> str:
    base = os.path.basename(path.replace("\\", "/").strip())
    for c in '<>:"/\\|?*':
        base = base.replace(c, "_")
    base = base.strip()
    return base or "wykres.png"


def extract_savefig_literal_paths(code: str) -> List[str]:
    """!
    @brief Kolejność wywołań savefig z dosłowną ścieżką w pierwszym argumencie (tylko typowe obrazy).
    """
    if not code.strip():
        return []
    out: List[str] = []
    for m in _SAVEFIG_LITERAL_RE.finditer(code):
        raw = m.group("p").strip()
        if not raw:
            continue
        cleaned = _sanitize_download_basename(raw)
        if os.path.splitext(cleaned.lower())[1] in _IMG_EXT:
            out.append(cleaned)
    return out


def apply_savefig_names_to_generated_outputs(
    generated_files: List[Dict[str, Any]],
    generated_images: List[Dict[str, Any]],
    generated_code: Optional[str],
) -> None:
    """!
    @brief Podmienia sztuczne nazwy obrazów na nazwy wynikające z kodu, odbudowuje generated_images.
    """
    paths = extract_savefig_literal_paths(generated_code or "")
    if not paths:
        return

    pi = 0
    for entry in generated_files:
        data = entry.get("data") or b""
        fn = entry.get("filename") or ""
        if not _is_image_entry(fn, data):
            continue
        if not is_container_style_filename(fn):
            continue
        if pi >= len(paths):
            break
        entry["filename"] = paths[pi]
        pi += 1

    generated_images[:] = [
        {"filename": e["filename"], "data": e["data"]}
        for e in generated_files
        if _is_image_entry(e.get("filename") or "", e.get("data") or b"")
    ]
