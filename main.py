from fastapi import FastAPI, File, UploadFile
from typing import List
import os
from fastapi import FastAPI

app = FastAPI(
    title="HieuMT DichVu AI",
    description="Nền tảng dịch vụ AI do HieuMT phát triển 🚀",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)  # tự động tạo thư mục nếu chưa có

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    return {"filename": file.filename, "size": len(content)}

@app.post("/api/upload-multi")
async def upload_multiple(files: List[UploadFile] = File(...)):
    file_infos = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        file_infos.append({"filename": file.filename, "size": len(content)})
    return {"uploaded_files": file_infos}
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionPipeline
import torch

# Khởi tạo model AI (chỉ chạy 1 lần khi server khởi động)
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

@app.post("/api/ai-generate")
async def ai_generate(prompt: str = "A fantasy landscape, detailed, 4k"):
    # Tạo ảnh AI từ prompt
    image = pipe(prompt).images[0]

    # Lưu ảnh vào thư mục uploads
    output_path = os.path.join(UPLOAD_DIR, "ai_result.png")
    image.save(output_path)

    return {"message": "AI image generated successfully!", "file_path": output_path}
from fastapi.responses import StreamingResponse
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionPipeline
import torch
import os

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Tạo pipeline AI chỉ 1 lần (khi khởi động server)
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

@app.post("/api/ai-generate")
async def ai_generate(prompt: str = "a cute cat astronaut floating in space, 4k"):
    # Sinh ảnh từ prompt
    image = pipe(prompt).images[0]

    # Lưu ảnh vào buffer (không cần ghi ra file)
    img_bytes = BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    # Trả ảnh trực tiếp về client
    return StreamingResponse(img_bytes, media_type="image/png")
