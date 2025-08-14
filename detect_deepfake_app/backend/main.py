from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from model.detect import predict_image
import os
from io import BytesIO
from PIL import Image
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "backend/temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Kiểm tra file hợp lệ
        if file is None:
            logger.error("No file provided")
            raise HTTPException(status_code=400, detail="No file provided")

        # Kiểm tra định dạng file
        logger.info("File content type: %s", file.content_type)
        if not file.content_type or not file.content_type.startswith("image/"):
            logger.error("Invalid file type: %s", file.content_type)
            raise HTTPException(status_code=400, detail=f"File phải là ảnh (JPG/PNG), received: {file.content_type}")

        # Đọc ảnh trực tiếp vào PIL Image
        content = await file.read()
        if not content:
            logger.error("Empty file content")
            raise HTTPException(status_code=400, detail="File is empty")
        img = Image.open(BytesIO(content))

        # Dự đoán
        logger.info("Predicting image: %s", file.filename)
        label, confidence = predict_image(img)

        return JSONResponse(content={
            "filename": file.filename,
            "label": label,
            "confidence": confidence  # Trả về dict chứa real và fake probabilities
        })

    except HTTPException as e:
        raise e  # Trả về mã trạng thái chính xác (400)
    except Exception as e:
        logger.error("Prediction error: %s", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)