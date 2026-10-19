"""PrusaSlicer CLI wrapper (headless slicing).

PrusaSlicer is not pip-installable and its Linux binary links against WebKitGTK, which
isn't present on this server. We therefore run the AppImage-extracted binary with a
locally-extracted set of the missing system libraries (see plan Phase 3 notes).

Environment overrides:
  PRUSA_BIN       path to the ``prusa-slicer`` ELF binary
  PRUSA_DATADIR   path to the bundled profiles (``resources``)
"""

from __future__ import annotations

import os
import subprocess

_APPDIR = "/tmp/opencode/squashfs-root"
_BIN = f"{_APPDIR}/usr/bin/bin/prusa-slicer"
_LIBS = "/tmp/opencode/slicer-libs/usr/lib/x86_64-linux-gnu"
_DATADIR = f"{_APPDIR}/usr/bin/resources"


def binary() -> str:
    return os.environ.get("PRUSA_BIN", _BIN)


def datadir() -> str:
    return os.environ.get("PRUSA_DATADIR", _DATADIR)


def slice_stl(
    stl_path: str,
    gcode_path: str,
    extra: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Slice ``stl_path`` to ``gcode_path``; returns the process with captured output."""
    env = os.environ.copy()
    bundled = f"{_APPDIR}/usr/bin/bin"
    ld = f"{bundled}:{_LIBS}"
    if env.get("LD_LIBRARY_PATH"):
        ld += f":{env['LD_LIBRARY_PATH']}"
    env["LD_LIBRARY_PATH"] = ld

    cmd = [binary(), "--datadir", datadir(), "--export-gcode", "--output", gcode_path, stl_path]
    cmd += extra or []
    return subprocess.run(cmd, capture_output=True, text=True, env=env)
