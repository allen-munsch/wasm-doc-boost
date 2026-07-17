"""
GLM-OCR backend service for wasm-doc-boost.
Loads zai-org/GLM-OCR via transformers, serves a single POST /ocr endpoint.

Usage:
    pip install -r requirements.txt
    python server.py --port 8765

First run will download the model (~2GB) from HuggingFace.
GPU recommended — CPU inference is extremely slow.
"""

import argparse
import base64
import io
import json
import os
import re
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import AutoProcessor, GlmOcrForConditionalGeneration

MODEL_ID = "zai-org/GLM-OCR"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = None
processor = None
layout_detector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor, layout_detector
    print(f"[glm-ocr] Loading {MODEL_ID} on {DEVICE}...")
    t0 = time.time()
    model = GlmOcrForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"[glm-ocr] Ready in {time.time() - t0:.1f}s")

    # Init layout detector on CPU (GPU is for the OCR model)
    print("[glm-ocr] Loading PP-DocLayoutV3 layout detector on CPU...")
    from glmocr.layout import PPDocLayoutDetector
    from glmocr.config import load_config as glmocr_load_config
    glm_cfg = glmocr_load_config()
    layout_cfg = glm_cfg.pipeline.layout
    layout_cfg.device = "cpu"
    layout_cfg.threshold = 0.3
    layout_detector = PPDocLayoutDetector(config=layout_cfg)
    layout_detector.start()
    print("[glm-ocr] Layout detector ready")

    yield

    print("[glm-ocr] Shutting down")
    layout_detector.stop()


app = FastAPI(title="GLM-OCR Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def detect_text_regions(pil_image: Image.Image) -> list[dict]:
    """Detect text regions using PP-DocLayoutV3 via glmocr SDK.

    Returns list of {bbox: [x1,y1,x2,y2]} with pixel coordinates.
    Keeps only regions with task_type in {'text', 'table', 'formula'}.
    """
    w, h = pil_image.size
    results = layout_detector.process([pil_image])
    # results is (pages, metadata); pages[0] is the list of region dicts
    regions = results[0][0] if results[0] else []

    out = []
    for r in regions:
        task = r.get("task_type", "text")
        if task not in ("text", "table", "formula"):
            continue
        # bbox_2d is [x1,y1,x2,y2] normalized to 0-1000
        bx1, by1, bx2, by2 = r["bbox_2d"]
        x1 = int(bx1 * w / 1000)
        y1 = int(by1 * h / 1000)
        x2 = int(bx2 * w / 1000)
        y2 = int(by2 * h / 1000)
        # Clamp to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        out.append({"bbox": [x1, y1, x2, y2]})
    return out


def _run_ocr(pil_image: Image.Image, prompt: str, max_tokens: int = 512) -> str:
    """Run GLM-OCR on a single image and return decoded text."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt_text, images=pil_image, return_tensors="pt").to(DEVICE)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            top_p=1.0,
            repetition_penalty=1.0,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    new_ids = generated[0, prompt_len:]
    return processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()


@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_ID}


@app.post("/ocr")
async def ocr(request: Request):
    """Accept image + prompt, return GLM-OCR JSON output.

    Supports two modes via `mode` field in request body:
      - "regions" (default): detect text regions, OCR each crop.
        Returns [{text, bbox: [x1,y1,x2,y2]}, ...]
      - "full": legacy single-pass OCR on the full image.
        Returns {text: "..."}
    """
    body = await request.json()
    image_b64 = body.get("image")
    prompt = body.get("prompt", "Text Recognition:")
    mode = body.get("mode", "regions")

    if not image_b64:
        raise HTTPException(status_code=400, detail="Missing 'image' (base64-encoded)")

    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    t0 = time.time()

    if mode == "full":
        output_text = _run_ocr(image, prompt, max_tokens=2048)
        output_text = re.sub(r'^```(?:json)?\s*\n?', '', output_text)
        output_text = re.sub(r'\n?```\s*$', '', output_text)
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError:
            parsed = {"text": output_text}
    else:
        regions = detect_text_regions(image)
        parsed = []
        for i, r in enumerate(regions):
            crop = image.crop((r["bbox"][0], r["bbox"][1], r["bbox"][2], r["bbox"][3]))
            text = _run_ocr(crop, prompt, max_tokens=512)
            text = re.sub(r'^```(?:json)?\s*\n?', '', text)
            text = re.sub(r'\n?```\s*$', '', text)
            parsed.append({"text": text.strip(), "bbox": r["bbox"]})

    elapsed = time.time() - t0

    # If regions mode, wrap in top-level dict for meta
    if isinstance(parsed, list):
        result = {"regions": parsed}
    else:
        result = parsed
    result["_meta"] = {
        "engine": f"GLM-OCR ({DEVICE})",
        "latency_ms": round(elapsed * 1000),
        "prompt_length": len(prompt),
    }
    return result


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    uvicorn.run(
        "backend.server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=["/app/backend"] if args.reload else None,
    )
