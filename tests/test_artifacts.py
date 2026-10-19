"""Tests for the artifact registry (dashboard manifest)."""

import json

from printable_surfaces.artifacts import register


def test_register_creates_and_is_idempotent(tmp_path):
    mf = tmp_path / "manifest.json"
    register("lattice", "output/x.stl", "stl", "X", manifest_path=str(mf))
    register("lattice", "output/x.stl", "stl", "X", manifest_path=str(mf))
    register("surface", "output/y.png", "image", "Y", manifest_path=str(mf))

    data = json.loads(mf.read_text())
    ids = [s["id"] for s in data["stages"]]
    assert ids == ["lattice", "surface"]

    lattice = data["stages"][0]
    assert len(lattice["artifacts"]) == 1
    assert lattice["artifacts"][0] == {"path": "output/x.stl", "kind": "stl", "label": "X"}


def test_register_rejects_unknown_stage(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        register("nonsense", "output/x.stl", "stl", "X", manifest_path=str(tmp_path / "m.json"))
