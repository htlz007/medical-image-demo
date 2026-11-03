"""
Medical Image Demo — ISIC 皮肤病变分割 API
==========================================
一个用 FastAPI 包起来的医学图像分割服务。

模型：SmallUNet（U-Net，base=16），在 ISIC 2018 上训练
权重：runs/isic/best_small_unet.pth

启动：
    python -m uvicorn main:app --reload
文档：
    http://127.0.0.1:8000/docs
"""

import base64
import io
import os
import sys

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

# 让 `from isic.xxx import` 能在任意工作目录下工作
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from isic.isic_models import build_model  # noqa: E402

# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
CKPT_PATH = os.path.join(ROOT, "runs", "isic", "best_small_unet.pth")
IMG_SIZE = 256
THRESHOLD = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------
# 启动时加载模型（只加载一次，不是每次请求都加载）
# --------------------------------------------------------------------------
app = FastAPI(
    title="Medical Image Demo",
    description="ISIC 2018 skin lesion segmentation — upload a dermoscopy image, get a mask.",
    version="0.1.0",
)

model = build_model("small_unet").to(DEVICE)
_ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(_ckpt["model"])
model.eval()

META = {
    "model": "SmallUNet",
    "dataset": "ISIC 2018",
    "epoch": _ckpt.get("epoch"),
    "val_dice": round(float(_ckpt.get("val_dice", 0.0)), 4),
    "device": DEVICE,
}


def preprocess(raw: bytes) -> torch.Tensor:
    """bytes -> (1, 3, 256, 256) float tensor in [0, 1]"""
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # (H,W,3) -> (3,H,W)
    return torch.from_numpy(arr).unsqueeze(0).to(DEVICE)


def to_png_b64(arr_uint8: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------
# 接口
# --------------------------------------------------------------------------
@app.get("/")
def health():
    """Health check — also tells you what model is loaded."""
    return {"status": "ok", **META}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload a dermoscopy image, get the lesion mask + statistics."""
    tensor = preprocess(await file.read())

    with torch.no_grad():
        prob = torch.sigmoid(model(tensor))

    mask = (prob > THRESHOLD).float()
    lesion_ratio = float(mask.mean().item())

    # 掩码（黑底白病变）
    mask_u8 = (mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
    # 概率图（灰度）
    prob_u8 = (prob[0, 0].cpu().numpy() * 255).astype(np.uint8)
    # 叠加图：原图上把病变区域刷红
    rgb = (tensor[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    overlay = rgb.copy()
    overlay[mask[0, 0].cpu().numpy() > 0] = [255, 0, 0]

    return {
        "filename": file.filename,
        "image_size": [IMG_SIZE, IMG_SIZE],
        "lesion_ratio": round(lesion_ratio, 4),
        "threshold": THRESHOLD,
        "mask_png_base64": to_png_b64(mask_u8),
        "prob_png_base64": to_png_b64(prob_u8),
        "overlay_png_base64": to_png_b64(overlay),
    }
