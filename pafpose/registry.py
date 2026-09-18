"""Load ``backends/backends.yaml`` and preset files, resolve part -> backend selections."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import yaml

from .schema import PARTS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = REPO_ROOT / "backends" / "backends.yaml"
DEFAULT_PRESET_DIR = REPO_ROOT / "configs"

COMMAND_PLACEHOLDERS = ("{video}", "{out}")


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class Upstream:
    repo: str
    commit: str


@dataclass(frozen=True)
class Backend:
    name: str
    display_name: str
    parts: tuple[str, ...]
    image: str
    gpu: bool
    command: str
    upstream: Upstream | None = None
    weights: tuple[str, ...] = field(default_factory=tuple)

    def supports(self, part: str) -> bool:
        return part in self.parts

    @property
    def needs_weights(self) -> bool:
        return bool(self.weights)


@dataclass(frozen=True)
class Selection:
    """Which backend serves which part."""

    body: str
    hand: str
    face: str

    def as_dict(self) -> dict[str, str]:
        return {"body": self.body, "hand": self.hand, "face": self.face}

    def grouped(self) -> dict[str, list[str]]:
        """backend name -> parts it serves, preserving body/hand/face order.

        A backend selected for several parts runs once.
        """
        groups: dict[str, list[str]] = {}
        for part, backend in self.as_dict().items():
            groups.setdefault(backend, []).append(part)
        return groups


def _parse_backend(name: str, raw: Mapping[str, object]) -> Backend:
    try:
        parts = tuple(raw["parts"])  # type: ignore[arg-type]
        image = str(raw["image"])
        command = str(raw["command"])
    except KeyError as exc:
        raise RegistryError(f"backend '{name}' missing required field {exc}") from None
    bad = [p for p in parts if p not in PARTS]
    if bad:
        raise RegistryError(f"backend '{name}' lists unknown parts {bad}; allowed {list(PARTS)}")
    if not parts:
        raise RegistryError(f"backend '{name}' lists no parts")
    for placeholder in COMMAND_PLACEHOLDERS:
        if placeholder not in command:
            raise RegistryError(f"backend '{name}' command lacks placeholder {placeholder}")
    upstream_raw = raw.get("upstream")
    upstream = None
    if upstream_raw:
        upstream = Upstream(repo=str(upstream_raw["repo"]), commit=str(upstream_raw["commit"]))  # type: ignore[index]
    weights = tuple(str(w) for w in (raw.get("weights") or []))
    if weights and "{weights}" not in command:
        raise RegistryError(f"backend '{name}' declares weights but command lacks {{weights}}")
    return Backend(
        name=name,
        display_name=str(raw.get("display_name", name)),
        parts=parts,
        image=image,
        gpu=bool(raw.get("gpu", True)),
        command=command,
        upstream=upstream,
        weights=weights,
    )


class Registry:
    def __init__(self, backends: Mapping[str, Backend], source: Path | None = None):
        self._backends = dict(backends)
        self.source = source

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Registry":
        resolved = Path(path or os.environ.get("PAFPOSE_REGISTRY") or DEFAULT_REGISTRY)
        if not resolved.is_file():
            raise RegistryError(f"registry file not found: {resolved}")
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        if raw.get("version") != 1:
            raise RegistryError(f"unsupported registry version {raw.get('version')!r} in {resolved}")
        entries = raw.get("backends") or {}
        backends = {name: _parse_backend(name, spec) for name, spec in entries.items()}
        return cls(backends, source=resolved)

    def names(self) -> list[str]:
        return list(self._backends)

    def get(self, name: str) -> Backend:
        try:
            return self._backends[name]
        except KeyError:
            raise RegistryError(f"unknown backend '{name}'; available: {', '.join(self.names())}") from None

    def __iter__(self):
        return iter(self._backends.values())

    def __len__(self) -> int:
        return len(self._backends)

    def for_part(self, part: str) -> list[Backend]:
        return [b for b in self._backends.values() if b.supports(part)]

    def resolve(
        self,
        body: str | None = None,
        hand: str | None = None,
        face: str | None = None,
        preset: str | Path | None = None,
    ) -> Selection:
        """Combine a preset (if any) with explicit overrides and check part support."""
        chosen: dict[str, str | None] = {"body": None, "hand": None, "face": None}
        if preset is not None:
            chosen.update(load_preset(preset))
        for part, value in (("body", body), ("hand", hand), ("face", face)):
            if value is not None:
                chosen[part] = value
        missing = [p for p, v in chosen.items() if v is None]
        if missing:
            raise RegistryError(f"no backend chosen for {missing}; pass --preset or --{'/--'.join(missing)}")
        for part, name in chosen.items():
            backend = self.get(str(name))
            if not backend.supports(part):
                raise RegistryError(
                    f"backend '{name}' does not produce '{part}' (it produces {list(backend.parts)}); "
                    f"candidates: {[b.name for b in self.for_part(part)]}"
                )
        return Selection(body=str(chosen["body"]), hand=str(chosen["hand"]), face=str(chosen["face"]))


def preset_path(name_or_path: str | Path) -> Path:
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate
    named = DEFAULT_PRESET_DIR / f"{name_or_path}.yaml"
    if named.is_file():
        return named
    available = sorted(p.stem for p in DEFAULT_PRESET_DIR.glob("*.yaml"))
    raise RegistryError(f"preset '{name_or_path}' not found; available: {available}")


def load_preset(name_or_path: str | Path) -> dict[str, str]:
    path = preset_path(name_or_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    unknown = [k for k in raw if k not in PARTS]
    if unknown:
        raise RegistryError(f"preset {path} has unknown keys {unknown}")
    return {k: str(v) for k, v in raw.items()}


def list_presets() -> list[str]:
    return sorted(p.stem for p in DEFAULT_PRESET_DIR.glob("*.yaml"))
