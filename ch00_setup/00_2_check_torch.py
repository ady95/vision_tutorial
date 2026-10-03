"""00-2. PyTorch가 GPU를 인식하는지 확인

실행:
    python ch00_setup/00_2_check_torch.py
"""
import torch

print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
