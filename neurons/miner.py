"""
TuneForge Miner — entry point.

Starts a TuneForgeMiner neuron that serves music generation
requests from validators on the TuneForge subnet.

Usage:
    python neurons/miner.py --env-file .env
    TF_WALLET_NAME=miner TF_WALLET_HOTKEY=default python neurons/miner.py
"""

import sys
from pathlib import Path


def _get_env_file() -> str | None:
    """Parse --env-file from argv before any heavy imports.

    Returns:
        Path to env file if specified, else None.
    """
    for i, arg in enumerate(sys.argv):
        if arg == "--env-file" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--env-file="):
            return arg.split("=", 1)[1]
    return None


def _load_env_file(env_path: str) -> None:
    """Load environment variables from a file.

    Supports simple KEY=VALUE format (one per line).
    Lines starting with # are ignored. Values may be quoted.

    Args:
        env_path: Path to the env file.
    """
    import os

    path = Path(env_path).expanduser()
    if not path.exists():
        print(f"Warning: env file not found: {path}", file=sys.stderr)
        return

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            os.environ[key] = value


def setup_logging() -> None:
    """Configure loguru logging for the miner process."""
    from tuneforge.settings import get_settings
    from tuneforge.utils.logging import setup_logging as _setup

    settings = get_settings()
    _setup(
        level=settings.log_level,
        log_dir=settings.log_dir,
        component_name="miner",
    )


def main() -> None:
    """Main entry point for the TuneForge miner."""
    # Parse env file before importing anything that reads env vars
    env_file = _get_env_file()
    if env_file:
        _load_env_file(env_file)

    setup_logging()

    from loguru import logger
    from tuneforge import VERSION
    from tuneforge.core.miner import TuneForgeMiner

    logger.info(f"Starting TuneForge Miner v{VERSION}")

    try:
        miner = TuneForgeMiner()
        miner.run()
    except KeyboardInterrupt:
        logger.info("Miner interrupted by user")
    except Exception as exc:
        logger.error(f"Miner exited with error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
