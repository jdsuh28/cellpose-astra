from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_astra_runtime_metadata_pins_supported_python_and_packages():
    setup_text = (ROOT / "setup.py").read_text()
    readme_text = (ROOT / "README.md").read_text()
    environment_text = (ROOT / "environment.yml").read_text()
    tox_text = (ROOT / "tox.ini").read_text()

    assert 'python_requires=">=3.11,<3.12"' in setup_text
    assert "'numpy==2.2.6'" in setup_text
    assert "'scipy==1.15.3'" in setup_text
    assert "'torch==2.10.0'" in setup_text
    assert "'torchvision==0.25.0'" in setup_text
    assert "'opencv-python-headless==4.13.0.92'" in setup_text
    assert 'url="https://github.com/jdsuh28/cellpose-astra"' in setup_text
    assert "python=3.11" in readme_text
    assert "Cellpose v4.2.1.1" in readme_text
    dino_revision = "6876159a11b4df116f30f667f8c9888617df0751"
    assert dino_revision in readme_text
    assert dino_revision in environment_text
    assert dino_revision in tox_text


def test_upstream_model_registry_includes_cpsam_v2_and_comparison_models():
    models_text = (ROOT / "cellpose/models.py").read_text()

    assert 'MODEL_NAMES = ["cpsam_v2", "cpdino", "cpdino-vitb", "cpsam"]' in models_text
