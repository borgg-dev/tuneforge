"""
TuneForge Validator entry point.

Usage:
    python neurons/validator.py --env-file .env.validator
"""

import sys
from pathlib import Path

from tuneforge.utils.config import get_env_file, load_env_or_exit
from tuneforge.utils.logging import setup_logging


def _get_env_file() -> Path:
    """Load environment file from --env-file argument."""
    return load_env_or_exit("validator")


def main() -> None:
    """Start the TuneForge validator neuron."""
    env_path = _get_env_file()

    # Import settings after env is loaded
    from tuneforge.settings import get_settings

    settings = get_settings()

    setup_logging(
        level=settings.log_level,
        log_dir=settings.log_dir,
        component_name="validator",
    )

    from loguru import logger

    logger.info(f"Starting TuneForge Validator (env={env_path})")
    logger.info(f"Network: netuid={settings.netuid}")
    logger.info(f"Wallet: {settings.wallet_name}/{settings.wallet_hotkey}")

    from tuneforge.core.validator import TuneForgeValidator

    try:
        validator = TuneForgeValidator(settings=settings)
        validator.run()
    except KeyboardInterrupt:
        logger.info("Validator interrupted — shutting down")
    except Exception as exc:
        logger.error(f"Validator crashed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
