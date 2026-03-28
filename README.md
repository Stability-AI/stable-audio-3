# Stable Audio 3

TAGLINE TBD

TBD Paper/blog links

Image(s) TBD

## Installation Guide

## Quick Start

## Features

## Flash attention (What I had to do)
.venv/bin/python -m ensurepip
uv add ninja
FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE FLASH_ATTENTION_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=8 .venv/bin/pip3 install flash-attn --no-build-isolation --no-binary flash-attn --force-reinstall --no-cache-dir --no-deps

# Create Wheel
MAX_JOBS=8 TORCH_CUDA_ARCH_LIST="9.0" pip wheel flash-attn --no-build-isolation -w ~/flash-attn-wheels/
## Docs