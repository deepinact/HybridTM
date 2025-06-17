# Installation

### Requirements
All the codes are tested in the following environment:
* Python 3.8
* PyTorch 2.2.0
* CUDA 12.1

### Base environment
refer [PT-V3](https://github.com/Pointcept/PointTransformerV3)
```
conda create -n pointcept python=3.8 -y
conda activate pointcept
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
# We use CUDA 11.8 and PyTorch 2.1.0 for our development of PTv3
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

cd libs/pointops
python setup.py install
cd ../..
```

### spconv
refer [spconv](https://github.com/traveller59/spconv)

``pip install spconv-cu120  # choose version match your local cuda version``

### Flash Attention
refer [flash-attention](https://github.com/Dao-AILab/flash-attention)

``pip install flash-attn --no-build-isolation``
