#!/usr/bin/env bash
set -euo pipefail

echo "=== Downloading TuneForge Models ==="

if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "Downloading MusicGen medium model..."
python3 -c "
from transformers import AutoProcessor, MusicgenForConditionalGeneration
print('Downloading MusicGen model...')
MusicgenForConditionalGeneration.from_pretrained('facebook/musicgen-medium')
AutoProcessor.from_pretrained('facebook/musicgen-medium')
print('MusicGen model downloaded.')
"

echo "Downloading CLAP model..."
python3 -c "
from transformers import ClapModel, ClapProcessor
print('Downloading CLAP model...')
ClapModel.from_pretrained('laion/larger_clap_music')
ClapProcessor.from_pretrained('laion/larger_clap_music')
print('CLAP model downloaded.')
"

echo "=== All models downloaded ==="
