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
ClapModel.from_pretrained('laion/clap-htsat-fused')
ClapProcessor.from_pretrained('laion/clap-htsat-fused')
print('CLAP model downloaded.')
"

# Optional: DiffRhythm v1.2 full model
# Uncomment to pre-download DiffRhythm weights (~4GB)
# Requires: git clone https://github.com/ASLP-lab/DiffRhythm ~/DiffRhythm
#
# echo "Downloading DiffRhythm full model..."
# python3 -c "
# from huggingface_hub import hf_hub_download
# import os
# save_dir = os.path.expanduser('~/DiffRhythm/pretrained')
# os.makedirs(save_dir, exist_ok=True)
# print('Downloading DiffRhythm CFM model...')
# hf_hub_download('ASLP-lab/DiffRhythm-1_2-full', filename='cfm_model.pt', local_dir=save_dir)
# print('Downloading DiffRhythm VAE...')
# hf_hub_download('ASLP-lab/DiffRhythm-vae', filename='vae_model.pt', local_dir=save_dir)
# print('DiffRhythm models downloaded.')
# "

echo "=== All models downloaded ==="
