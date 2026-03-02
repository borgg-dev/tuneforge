"""
Configuration parsing utilities for TuneForge Subnet.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv


def get_env_file() -> Path | None:
    """
    Get env file path from --env-file command line argument.

    Returns None if not specified.
    """
    for i, arg in enumerate(sys.argv):
        if arg == "--env-file" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1])
        if arg.startswith("--env-file="):
            return Path(arg.split("=", 1)[1])
    return None


def load_env_or_exit(role: str) -> Path:
    """
    Load environment file or exit with error message.

    Args:
        role: The neuron role (miner or validator) for error messages.

    Returns:
        Path to the loaded env file.
    """
    env_file = get_env_file()
    if env_file is None:
        sys.stderr.write(
            f"ERROR: --env-file is required. Example: --env-file .env.{role}\n"
        )
        sys.stderr.write(f"Copy .env.{role}.example to .env.{role} and configure it.\n")
        sys.exit(1)
    if env_file.exists():
        load_dotenv(env_file)
    else:
        sys.stderr.write(f"ERROR: Config file not found: {env_file}\n")
        sys.exit(1)
    return env_file
