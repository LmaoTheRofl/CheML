from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REQUIRED_TOOL_NAMES = (
    "pymupdf",
    "pymupdf_layout",
    "marker",
    "ocr",
    "molscribe",
    "rdkit",
    "codex",
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _project_ocr_command() -> str | None:
    root = _project_root()
    prefix = root / "runs" / "tools" / "tesseract"
    traineddata = prefix / "tessdata" / "eng.traineddata"
    if not traineddata.is_file() or _which("tesseract") is None:
        return None
    return shlex.join(
        [
            sys.executable,
            "-m",
            "chemx.tesseract_ocr",
            "--tessdata-prefix",
            str(prefix),
            "{image}",
        ]
    )


def _project_molscribe_command() -> str | None:
    root = _project_root()
    python = root / "runs" / "tools" / "molscribe-py39" / "bin" / "python"
    adapter = root / "scripts" / "molscribe_predict.py"
    weights = root / "swin_base_char_aux_1m680k.pth"
    if not all(path.is_file() for path in (python, adapter, weights)):
        return None
    return shlex.join(
        [str(python), str(adapter), "--model", str(weights), "{image}"]
    )


class ToolchainError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolStatus:
    name: str
    available: bool
    detail: str
    command: str | None = None
    version: str | None = None


def _module_status(name: str, module: str) -> ToolStatus:
    spec = importlib.util.find_spec(module)
    if spec is None:
        return ToolStatus(name, False, f"missing Python module: {module}")
    version = None
    try:
        imported = __import__(module)
        version = str(getattr(imported, "__version__", "")) or None
    except Exception:
        version = None
    return ToolStatus(name, True, f"Python module importable: {module}", version=version)


def _which(command: str) -> str | None:
    found = shutil.which(command)
    if found:
        return found
    scripts = Path(sys.executable).resolve().parent
    suffixes = ["", ".exe", ".cmd", ".bat"] if os.name == "nt" else [""]
    for suffix in suffixes:
        candidate = scripts / f"{command}{suffix}"
        if candidate.is_file():
            return str(candidate)
    return None


def _split_command(command: str) -> list[str]:
    parts = shlex.split(command)
    if os.name != "nt" or len(parts) < 2 or Path(parts[0]).is_file():
        return parts
    for index in range(1, min(len(parts), 5)):
        candidate = " ".join(parts[: index + 1])
        if Path(candidate).is_file() or _which(candidate):
            return [candidate, *parts[index + 1 :]]
    return parts


def _command_status(name: str, command: str | None, fallback: str | None = None) -> ToolStatus:
    configured = command or fallback
    if not configured:
        return ToolStatus(name, False, "missing command configuration")
    parts = _split_command(configured)
    if not parts:
        return ToolStatus(name, False, "empty command configuration")
    executable = _which(parts[0]) or (parts[0] if Path(parts[0]).is_file() else None)
    if executable is None:
        return ToolStatus(name, False, f"missing executable: {parts[0]}", command=configured)
    return ToolStatus(name, True, f"executable found: {executable}", command=configured)


class FullStackToolchain:
    """Mandatory parser toolchain for production ChemX extraction."""

    def __init__(
        self,
        *,
        marker_command: str = (
            "marker_single --disable_ocr --disable_multiprocessing --disable_tqdm"
        ),
        ocr_command: str | None = None,
        molscribe_command: str | None = None,
        codex_command: str = "codex",
    ) -> None:
        self.marker_command = marker_command
        self.ocr_command_value = (
            ocr_command or os.environ.get("CHEMX_OCR_COMMAND") or _project_ocr_command()
        )
        self.molscribe_command_value = molscribe_command or os.environ.get(
            "CHEMX_MOLSCRIBE_COMMAND"
        ) or _project_molscribe_command()
        self.codex_command = codex_command

    def check(self) -> list[ToolStatus]:
        return [
            _module_status("pymupdf", "fitz"),
            _module_status("pymupdf_layout", "pymupdf.layout"),
            _command_status("marker", self.marker_command),
            _command_status("ocr", self.ocr_command_value, fallback="tesseract"),
            _command_status("molscribe", self.molscribe_command_value),
            _module_status("rdkit", "rdkit"),
            _command_status("codex", self.codex_command),
        ]

    def require(self) -> list[ToolStatus]:
        statuses = self.check()
        missing = [status for status in statuses if not status.available]
        if missing:
            details = "; ".join(f"{status.name}: {status.detail}" for status in missing)
            raise ToolchainError(f"mandatory ChemX parser toolchain is incomplete: {details}")
        return statuses

    def marker_executable(self) -> str:
        executable = _which(_split_command(self.marker_command)[0])
        if executable is None:
            raise ToolchainError(f"missing Marker executable: {self.marker_command}")
        return executable

    def ocr_command(self, image: Path) -> list[str]:
        configured = self.ocr_command_value
        if configured:
            parts = _split_command(configured)
            formatted = [part.format(image=str(image)) for part in parts]
            return formatted if "{image}" in configured else formatted + [str(image)]
        return ["tesseract", str(image), "stdout", "-l", "eng"]

    def molscribe_command(self, image: Path) -> list[str]:
        configured = self.molscribe_command_value
        if not configured:
            raise ToolchainError(
                "MolScribe runtime was not found; install the project-local runtime or set "
                "CHEMX_MOLSCRIBE_COMMAND"
            )
        parts = _split_command(configured)
        formatted = [part.format(image=str(image)) for part in parts]
        return formatted if "{image}" in configured else formatted + [str(image)]


def write_tool_manifest(
    path: Path,
    statuses: list[ToolStatus],
    *,
    artifacts: dict[str, str | None] | None = None,
) -> Path:
    payload = {
        "schema_version": "1.0",
        "python": sys.version,
        "tools": [asdict(status) for status in statuses],
        "artifacts": artifacts or {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_command(command: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
