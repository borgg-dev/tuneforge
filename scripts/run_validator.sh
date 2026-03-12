#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

ENV_FILE="${1:-.env.validator}"

if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: Environment file not found: $ENV_FILE"
    echo "Copy .env.validator.example to $ENV_FILE and configure it."
    exit 1
fi

if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "Starting TuneForge Validator with $ENV_FILE"
exec python3 -m neurons.validator --env-file "$ENV_FILE"
