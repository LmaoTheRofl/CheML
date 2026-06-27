from __future__ import annotations

import json
import os
import shlex
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from chemx.normalize import canonicalize_smiles


@dataclass(frozen=True)
class ResolvedChemical:
    query: str
    smiles: str
    source: str


def _get_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "chemx-article-parser/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def resolve_name(name: str, timeout: float = 8.0) -> ResolvedChemical | None:
    """Resolve a chemical name with PubChem first and OPSIN as a fallback."""
    encoded = urllib.parse.quote(name, safe="")
    try:
        data = _get_json(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded}/property/CanonicalSMILES/JSON",
            timeout,
        )
        smiles = data["PropertyTable"]["Properties"][0]["ConnectivitySMILES"]
        return ResolvedChemical(name, canonicalize_smiles(smiles), "pubchem")
    except (KeyError, OSError, ValueError):
        pass
    try:
        data = _get_json(f"https://opsin.ch.cam.ac.uk/opsin/{encoded}.json", timeout)
        return ResolvedChemical(name, canonicalize_smiles(data["smiles"]), "opsin")
    except (KeyError, OSError, ValueError):
        return None


def molscribe_image(image: Path, timeout: float = 120.0) -> str | None:
    """Run a separately installed MolScribe-compatible CLI and canonicalize its stdout."""
    configured = os.environ.get("CHEMX_MOLSCRIBE_COMMAND")
    if not configured:
        return None
    command = [*shlex.split(configured), str(image)]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    candidate = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
    return canonicalize_smiles(candidate) if candidate else None


def canonicalize_smiles_required(value: object) -> tuple[str | None, bool]:
    """Canonicalize with RDKit; raise if RDKit is unavailable."""
    if value is None:
        return None, True
    raw = str(value).strip()
    if not raw:
        return raw, True
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is required for ChemX SMILES canonicalization") from exc
    molecule = Chem.MolFromSmiles(raw)
    if molecule is None:
        return raw, False
    return Chem.MolToSmiles(molecule, canonical=True), True
