import os
import torch
import yaml
from torchvision import transforms
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .efficientvit.efficient_vit import EfficientViT

config_path = os.path.join(os.path.dirname(__file__), "efficientvit", "configs", "architecture.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Tạm thời ghi đè num_classes về 1 để khớp với checkpoint
config['model']['num-classes'] = 1

_model = None
def get_model():
    global _model
    if _model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = os.path.join(os.path.dirname(__file__), "efficient_vit.pth")
        logger.info(f"Loading model from {checkpoint_path} on {device}")
        if not os.path.exists(checkpoint_path):
            logger.error("Checkpoint not found: %s", checkpoint_path)
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        _model = EfficientViT(config, checkpoint_path=checkpoint_path).to(device)
        _model.eval()
    return _model

_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Đảm bảo resize về 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa ảnh
])

def predict_image(image: Image.Image) -> tuple[str, dict]:
    try:
        if not isinstance(image, Image.Image):
            logger.error("Invalid input: Expected PIL.Image.Image, got %s", type(image))
            raise ValueError("Đầu vào phải là đối tượng PIL.Image")

        # Transform ảnh
        logger.info("Transforming image with size %s", image.size)
        input_tensor = _transform(image).unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info("Input tensor shape: %s", input_tensor.shape)

        model = get_model()
        logger.info("Running inference")
        with torch.no_grad():
            output = model(input_tensor)  # Output shape: [batch_size, 1]
            logger.info(f"Raw output before sigmoid: {output.item()}")  # Log giá trị thô
            probability = torch.sigmoid(output)[0].item()  # Xác suất cho "real"
            logger.info(f"Probability (Real): {probability:.4f}, Fake: {1 - probability:.4f}")
            real_prob = probability
            fake_prob = 1 - probability  # Xác suất cho "fake"

        # Thử nghiệm ngưỡng phân loại
        threshold = 0.5  # Có thể điều chỉnh (ví dụ: 0.3 hoặc 0.7)
        label = "real" if real_prob > threshold else "fake"
        logger.info(f"Prediction with threshold {threshold}: {label}, Real: {real_prob:.4f}, Fake: {fake_prob:.4f}")
        return label, {"real": real_prob, "fake": fake_prob}

    except Exception as e:
        logger.error("Error during prediction: %s", str(e))
        raise RuntimeError(f"Lỗi khi dự đoán ảnh: {str(e)}")