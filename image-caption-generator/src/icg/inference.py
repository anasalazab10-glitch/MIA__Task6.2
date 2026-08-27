import os
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

from .vocab import load_vocab
from .model import ImageCaptioningModel
from .features import ResNet50FeatureExtractor

DEFAULT_REPO_ID = "Anas1010/flickr8k-image-caption-generator"

def ensure_artifact(local_path: str, repo_id: str, filename: str) -> str:
    # If file exists locally, use it; otherwise download from HF into cache and return cached path.
    if local_path and os.path.exists(local_path):
        return local_path
    return hf_hub_download(repo_id=repo_id, filename=filename)

class CaptionGenerator:
    def __init__(
        self,
        vocab_path: str = "",
        checkpoint_path: str = "",
        repo_id: str = DEFAULT_REPO_ID,
        device: str | None = None,
        feature_dim: int = 2048,
        embed_size: int = 256,
        hidden_size: int = 256,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        vocab_path = ensure_artifact(vocab_path, repo_id, "vocab.json")
        checkpoint_path = ensure_artifact(checkpoint_path, repo_id, "best_model.pt")

        self.vocab = load_vocab(vocab_path)

        self.model = ImageCaptioningModel(
            feature_dim=feature_dim,
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=self.vocab.size,
            pad_idx=self.vocab.pad_idx,
        ).to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.extractor = ResNet50FeatureExtractor(device=self.device)

    @torch.no_grad()
    def generate(self, image: Image.Image, max_len: int | None = None) -> str:
        max_len = max_len or self.vocab.max_len
        feat = self.extractor.extract(image).unsqueeze(0).to(self.device)  # (1,2048)

        # init hidden from image
        img_h = torch.relu(self.model.feature_proj(feat))  # (1, hidden)
        h = img_h.unsqueeze(0).contiguous()
        c = torch.zeros_like(h)

        cur = torch.tensor([[self.vocab.start_idx]], dtype=torch.long, device=self.device)

        out_tokens = []
        for _ in range(max_len - 1):
            emb = self.model.embedding(cur)            # (1,1,embed)
            out, (h, c) = self.model.lstm(emb, (h, c))
            logits = self.model.fc(out.squeeze(1))     # (1,V)
            next_id = int(torch.argmax(logits, dim=-1).item())

            if next_id == self.vocab.end_idx:
                break

            out_tokens.append(self.vocab.idx2word[next_id])
            cur = torch.tensor([[next_id]], dtype=torch.long, device=self.device)

        return " ".join(out_tokens).strip()
