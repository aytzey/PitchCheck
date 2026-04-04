import subprocess
import os


def test_docker_compose_config_valid():
    """docker compose config should validate without errors."""
    result = subprocess.run(
        ["docker", "compose", "config", "--quiet"],
        cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"docker compose config failed: {result.stderr}"
