from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_docker_build_downloads_required_parser_models() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "ARG DOWNLOAD_MODELS=1" in dockerfile
    assert "ARG CHEMX_MODEL_DOWNLOAD_READ_TIMEOUT_SECONDS=180" in dockerfile
    assert "scripts/download_datalab_cache.py --include-weights" in dockerfile
    assert "scripts/download_molscribe_model.py" in dockerfile
    assert "uv venv --seed --python 3.10 /opt/molscribe-venv" in dockerfile
    assert "/opt/chemx-models/molscribe/swin_base_char_aux_1m680k.pth" in dockerfile
    assert "/workspace/swin_base_char_aux_1m680k.pth" not in dockerfile

    assert 'DOWNLOAD_MODELS: "${DOWNLOAD_MODELS:-1}"' in compose
    assert (
        'CHEMX_MODEL_DOWNLOAD_READ_TIMEOUT_SECONDS: '
        '"${CHEMX_MODEL_DOWNLOAD_READ_TIMEOUT_SECONDS:-180}"'
    ) in compose
    assert "/opt/chemx-models/molscribe/swin_base_char_aux_1m680k.pth" in compose
    assert "/workspace/swin_base_char_aux_1m680k.pth" not in compose
    assert "chemx-cache:/opt/chemx-cache" not in compose
    assert "ollama/ollama:${OLLAMA_IMAGE_TAG:-latest}" in compose
    assert "ollama-models:/root/.ollama" in compose
    assert 'entrypoint: ["ollama"]' in compose
    assert 'command: ["pull", "${OLLAMA_MODEL:-lukaspetrik/gemma3-tools:27b}"]' in compose
    assert "ollama-adapter:" in compose
    assert "python\", \"-m\", \"chemx.ollama_adapter" in compose
    assert 'CHEMX_OLLAMA_ADAPTER_URL: "http://ollama-adapter:11434"' in compose
    assert 'OLLAMA_UPSTREAM_URL: "ollama:11434"' in compose
