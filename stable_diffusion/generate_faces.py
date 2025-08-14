import os
import random
from stable_diffusion import StableDiffusionXLPipeline
import torch
from PIL import Image
from tqdm import tqdm

# face_prompts.py

male_prompts = [
    "Ultra-realistic portrait of a young man in his 20s, looking directly at the camera, sharp facial features, clear skin, cinematic lighting, shallow depth of field, studio background, 85mm lens, f1.4, HDR, high resolution",
    "A rugged middle-aged man with a beard, realistic skin texture, moody lighting, detailed facial structure, close-up headshot, neutral background, sharp focus, 50mm lens, ultra-high resolution",
    "Close-up of a handsome man with blue eyes and brown hair, perfect skin, ultra photorealistic style, soft studio lighting, professional headshot, shallow DOF, 8k photo",
    "Realistic portrait of a man with dark skin tone, glowing skin under softbox light, studio photo, ultra sharp, fashion style",
    "Sharp image of a man smiling subtly, casual hairstyle, warm lighting, natural skin tones, detailed eyebrows, Nikon D850, 35mm lens"
]

female_prompts = [
    "Ultra-realistic portrait of a beautiful woman with long hair, natural makeup, soft expression, cinematic lighting, studio headshot, shallow DOF, glowing skin",
    "Photorealistic close-up of an Asian woman, delicate features, soft skin texture, fashion editorial style, muted tones, soft light from left side",
    "A mature woman with freckles, natural beauty, subtle smile, high detail, professional lighting, minimal retouching",
    "Portrait of a cheerful woman in natural light, flawless skin, subtle shadows, expressive eyes, clean image",
    "Hyper-realistic image of a blonde woman in her 30s, lightly styled hair, minimal makeup, soft shadows, 8k resolution"
]

# ========== CẤU HÌNH ==========
OUTPUT_DIR = "output_faces"
NUM_IMAGES = 6000
RESOLUTIONS = [(512, 512), (768, 768), (1024, 1024)]
USE_LORA = False
LORA_PATH = "lora_checkpoints/realvisxlV50_SDXL_v50LightningBakedvae.safetensors"

# ========== TẢI MODEL ==========
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

if USE_LORA:
    pipe.load_lora_weights(LORA_PATH)

pipe.enable_model_cpu_offload()

# ========== TẠO THƯ MỤC ==========
for gender in ["male", "female"]:
    for res in RESOLUTIONS:
        folder = os.path.join(OUTPUT_DIR, gender, f"{res[0]}x{res[1]}")
        os.makedirs(folder, exist_ok=True)

# ========== VÒNG LẶP GEN ẢNH ==========
for i in tqdm(range(NUM_IMAGES), desc="Generating faces"):
    gender = "male" if i < NUM_IMAGES // 2 else "female"
    prompt = random.choice(male_prompts if gender == "male" else female_prompts)
    width, height = random.choice(RESOLUTIONS)
    
    image = pipe(
        prompt,
        width=width,
        height=height,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    save_folder = os.path.join(OUTPUT_DIR, gender, f"{width}x{height}")
    image_path = os.path.join(save_folder, f"{gender}_{i:04d}_{width}x{height}.png")
    image.save(image_path)
