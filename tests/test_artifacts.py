"""Tests for the artifact registry (dashboard manifest)."""

import json

from printable_surfaces.artifacts import register, remove_artifact


def test_register_creates_folder_and_is_idempotent(tmp_path):
    mf = tmp_path / "manifest.json"
    register("lattice", "output/x.stl", "stl", "X", folder="Batch A", manifest_path=str(mf))
    register("lattice", "output/x.stl", "stl", "X", folder="Batch A", manifest_path=str(mf))
    register("surface", "output/y.png", "image", "Y", folder="Batch A", manifest_path=str(mf))

    data = json.loads(mf.read_text())
    assert [f["name"] for f in data["folders"]] == ["Batch A"]
    arts = data["folders"][0]["artifacts"]
    assert len(arts) == 2
    assert arts[0] == {"path": "output/x.stl", "kind": "stl", "label": "X", "stage": "lattice"}


def test_register_new_folder_goes_first(tmp_path):
    mf = tmp_path / "manifest.json"
    register("lattice", "a.stl", "stl", "A", folder="Old", manifest_path=str(mf))
    register("lattice", "b.stl", "stl", "B", folder="New", manifest_path=str(mf))

    data = json.loads(mf.read_text())
    assert [f["name"] for f in data["folders"]] == ["New", "Old"]


def test_register_defaults_folder_to_stage(tmp_path):
    mf = tmp_path / "manifest.json"
    register("lattice", "c.stl", "stl", "C", manifest_path=str(mf))
    data = json.loads(mf.read_text())
    assert data["folders"][0]["name"] == "lattice"


def test_register_rejects_unknown_stage(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        register("nonsense", "x.stl", "stl", "X", manifest_path=str(tmp_path / "m.json"))


def test_remove_artifact_and_empty_folder(tmp_path):
    mf = tmp_path / "manifest.json"
    register("lattice", "output/a.stl", "stl", "A", folder="Batch", manifest_path=str(mf))
    register("lattice", "output/b.stl", "stl", "B", folder="Batch", manifest_path=str(mf))
    remove_artifact("output/a.stl", manifest_path=str(mf))

    data = json.loads(mf.read_text())
    paths = [a["path"] for a in data["folders"][0]["artifacts"]]
    assert paths == ["output/b.stl"]

    remove_artifact("output/b.stl", manifest_path=str(mf))
    assert json.loads(mf.read_text())["folders"] == []
