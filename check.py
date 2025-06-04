import torch
import subprocess

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
    print("nvidia-smi:\n", result.stdout)
except subprocess.CalledProcessError as e:
    print("nvidia-smi failed:\n", e.stderr)