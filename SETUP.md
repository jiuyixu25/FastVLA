# Setup Instructions

## Set Up Conda Environment

```bash
# Create and activate conda environment
conda create -n fastvla python=3.10 -y
conda activate fastvla

# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio

# Clone the FastVLA repo and pip install to download dependencies
git clone https://github.com/jiuyixu25/FastVLA.git
cd FastVLA
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```