import torch
checkpoint = torch.load("D:/DATN/detect_deepfake_app/backend/model/efficient_vit.pth")
print("Checkpoint keys and shapes:")
for key, value in checkpoint.items():
    print(f"{key}: {value.shape}")