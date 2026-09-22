from pathlib import Path

import pytest

from pafpose import registry as reg

FIRST_CLASS = {"sam3dbody", "pear", "wilor", "teaser", "mediapipe"}


def test_bundled_registry_loads():
    registry = reg.Registry.load()
    assert set(registry.names()) == FIRST_CLASS
    assert registry.get("pear").parts == ("body", "hand", "face")
    assert registry.get("mediapipe").gpu is False
    assert registry.get("mediapipe").upstream is None


def test_every_bundled_preset_resolves():
    registry = reg.Registry.load()
    assert set(reg.list_presets()) == {"accuracy", "balanced", "speed"}
    for name in reg.list_presets():
        registry.resolve(preset=name)


def test_resolve_rejects_backend_that_lacks_part():
    registry = reg.Registry.load()
    with pytest.raises(reg.RegistryError, match="does not produce 'face'"):
        registry.resolve(body="pear", hand="pear", face="wilor")


def test_resolve_requires_every_part():
    registry = reg.Registry.load()
    with pytest.raises(reg.RegistryError, match="no backend chosen"):
        registry.resolve(body="pear")


def test_explicit_flags_override_preset():
    registry = reg.Registry.load()
    selection = registry.resolve(preset="speed", hand="wilor")
    assert selection.as_dict() == {"body": "pear", "hand": "wilor", "face": "pear"}


def test_grouped_runs_shared_backend_once():
    selection = reg.Selection(body="sam3dbody", hand="sam3dbody", face="pear")
    assert selection.grouped() == {"sam3dbody": ["body", "hand"], "pear": ["face"]}


def test_invalid_registry_reports_missing_placeholder(tmp_path: Path):
    bad = tmp_path / "backends.yaml"
    bad.write_text(
        "version: 1\nbackends:\n  x:\n    parts: [body]\n    image: x:1\n    command: python a.py\n",
        encoding="utf-8",
    )
    with pytest.raises(reg.RegistryError, match="lacks placeholder"):
        reg.Registry.load(bad)
