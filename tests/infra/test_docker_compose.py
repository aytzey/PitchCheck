import subprocess
import os


def test_docker_compose_config_valid():
    """docker compose config should validate without errors."""
    env = {**os.environ, "SCORE_API_KEY": os.environ.get("SCORE_API_KEY", "test-compose-secret")}
    result = subprocess.run(
        ["docker", "compose", "config", "--quiet"],
        cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"docker compose config failed: {result.stderr}"
