# CraftReplicator
Digital Twins with YOLO and PyTorch3D



## Pytorch3D MacOS installation steps:

1. First, create and activate a virtual environment using conda (as recommended in the [docs](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)):

```bash
conda create -n pytorch3d python=3.9
conda activate pytorch3d
```

2. For Mac M2 (Apple Silicon), you need to install PyTorch and torchvision. 

```bash
conda install pytorch torchvision -c pytorch
```

3. Install ioPath dependency:

```bash
conda install -c iopath iopath
```

4. For Mac installation, it's necessary to set specific environment variables before installing PyTorch3D. The documentation specifically mentions this for macOS:

```bash
MACOSX_DEPLOYMENT_TARGET=10.14 CC=clang CXX=clang++ pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

5. To run demos or the code later, you'll need these additional packages:

```bash
conda install jupyter nb_conda
pip install scikit-image matplotlib imageio plotly opencv-python
```
