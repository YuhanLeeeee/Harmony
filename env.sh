#!/usr/bin/env bash
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip install ./requirements/causal_conv1d-1.4.0+cu122torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
pip install ./requirements/mamba_ssm-2.2.2+cu122torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
pip install -r ./requirements/requirements.txt