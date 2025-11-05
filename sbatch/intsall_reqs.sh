# pip uninstall -y mmpose mmcv-full mmcv mmengine numpy xtcocotools pycocotools

# # Install a stable numpy version
# pip install numpy==1.23.5

# # Install PyTorch that you already have (check first)
# python -c "import torch; print(torch.__version__)"


# Install MMCV for CUDA 12.8 + PyTorch 2.9 (via the official OpenMMLab index)
pip install "mmcv-full==1.7.2" -f https://download.openmmlab.com/mmcv/dist/cu128/torch2.9/index.html

# Install the matching MMPose version
pip install "mmpose==1.2.0"

# Install xtcocotools (required for COCO format keypoint handling)
pip install "xtcocotools==1.12"
