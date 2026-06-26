from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_astra_runtime_metadata_pins_supported_python_and_packages():
    setup_text = (ROOT / "setup.py").read_text()
    readme_text = (ROOT / "README.md").read_text()

    assert 'python_requires=">=3.10,<3.11"' in setup_text
    assert "'numpy>=1.20.0,<2'" in setup_text
    assert "'torch==2.2.2'" in setup_text
    assert "'torchvision==0.17.2'" in setup_text
    assert "'opencv-python-headless<4.12'" in setup_text
    assert "python=3.10" in readme_text
    assert "v4.1.1+astra.3" in readme_text
