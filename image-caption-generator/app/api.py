import io
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

from icg.inference_attn import AttentionCaptionGenerator


def create_app() -> FastAPI:
    app = FastAPI(title="Image Caption Generator (Attention)")

    # Optional local paths; if empty, it auto-downloads from HF repo_id.
    vocab_path = os.getenv("ICG_VOCAB_PATH", "")
    ckpt_path  = os.getenv("ICG_CKPT_PATH", "")
    repo_id    = os.getenv("ICG_REPO_ID", "Anas1010/flickr8k-image-caption-generator-attn")
    device     = os.getenv("ICG_DEVICE", "cpu")

    generator = AttentionCaptionGenerator(
        vocab_path=vocab_path,
        checkpoint_path=ckpt_path,
        repo_id=repo_id,
        device=device,
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/caption")
    async def caption_image(file: UploadFile = File(...)):
        if file.content_type is None or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Please upload an image file.")

        data = await file.read()
        try:
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image.")

        caption = generator.generate(img)
        return {"filename": file.filename, "caption": caption}

    return app


app = create_app()
