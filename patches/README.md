# Transformers 4.48.0 Bark CUDA Compatibility Patch

This directory contains a targeted compatibility patch for Bark
generation using NVIDIA CUDA devices.

## Target version

- Transformers: 4.48.0
- Model: suno/bark-small
- Tested GPU: NVIDIA RTX 5090 Laptop GPU

## Changes

The patch contains three device-placement operations:

1. Move semantic history tensors to the semantic generation device.
2. Move coarse history tensors to the semantic generation device.
3. Move fine history tensors to the fine input device.

These changes address tensor-device mismatches during Bark generation.

## Patch file

transformers-4.48.0-bark-cuda.patch

The patch was generated against the original Transformers 4.48.0
source distributed in its Python wheel.

## Applying the patch

First locate the installed Transformers package.

Example:

/home/mtrx1610/miniconda3/envs/genai_fastapi_p311/lib/python3.11/site-packages

Change to the site-packages directory:

cd /home/mtrx1610/miniconda3/envs/genai_fastapi_p311/lib/python3.11/site-packages

Back up the original file:

cp transformers/models/bark/modeling_bark.py \
   transformers/models/bark/modeling_bark.py.backup

Return to the repository:

cd ~/github/bark-rtx-gpu

Check patch applicability:

patch --dry-run -p1 \
  -d /home/mtrx1610/miniconda3/envs/genai_fastapi_p311/lib/python3.11/site-packages \
  < patches/transformers-4.48.0-bark-cuda.patch

If the dry run succeeds, apply:

patch -p1 \
  -d /home/mtrx1610/miniconda3/envs/genai_fastapi_p311/lib/python3.11/site-packages \
  < patches/transformers-4.48.0-bark-cuda.patch

## Reverting

Restore the original file from the backup:

cp /home/mtrx1610/miniconda3/envs/genai_fastapi_p311/lib/python3.11/site-packages/transformers/models/bark/modeling_bark.py.backup \
   /home/mtrx1610/miniconda3/envs/genai_fastapi_p311/lib/python3.11/site-packages/transformers/models/bark/modeling_bark.py

## Important

- This patch targets Transformers 4.48.0.
- Do not apply it repeatedly.
- Do not assume it applies cleanly to other Transformers versions.
- Back up the original source before applying.
- The patch modifies installed third-party source code.
- These modifications are not official upstream Transformers fixes.

The patch was independently tested against an unmodified
Transformers 4.48.0 source file.
