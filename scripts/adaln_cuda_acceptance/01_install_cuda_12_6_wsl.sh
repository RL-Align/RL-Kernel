#!/usr/bin/env bash
set -euo pipefail

grep -qi microsoft /proc/version || { echo "WSL2 is required" >&2; exit 1; }
if [[ -x /usr/local/cuda-12.6/bin/nvcc ]] &&
   /usr/local/cuda-12.6/bin/nvcc --version | grep -q 'release 12\.6'; then
  echo "CUDA Toolkit 12.6 is already installed"
  exit 0
fi

read -r -p "Install the large cuda-toolkit-12-6 package (no Linux driver)? [y/N] " reply
[[ "$reply" =~ ^[Yy]$ ]] || exit 0

deb=$(mktemp --suffix=.deb)
trap 'rm -f "$deb"' EXIT
curl -fL   https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb   -o "$deb"
sudo dpkg -i "$deb"
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6
