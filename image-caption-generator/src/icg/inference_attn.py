import os
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

from .vocab import load_vocab
from .attn_model import AttnCaptionModel
from .spatial_features import ResNet50SpatialFeatureExtractor

DEFAULT_REPO_ID = "Anas1010/flickr8k-image-caption-generator-attn"


def ensure_artifact(local_path: str, repo_id: str, filename: str) -> str:
    if local_path and os.path.exists(local_path):
        return local_path
    return hf_hub_download(repo_id=repo_id, filename=filename)


class AttentionCaptionGenerator:
    def __init__(
        self,
        vocab_path: str = "",
        checkpoint_path: str = "",
        repo_id: str = DEFAULT_REPO_ID,
        device: str | None = None,
        embed_dim: int = 256,
        dec_dim: int = 512,
        attn_dim: int = 256,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        vocab_path = ensure_artifact(vocab_path, repo_id, "vocab.json")
        checkpoint_path = ensure_artifact(checkpoint_path, repo_id, "best_model_attn.pt")

        self.vocab = load_vocab(vocab_path)

        self.model = AttnCaptionModel(
            vocab_size=self.vocab.size,
            pad_idx=self.vocab.pad_idx,
            embed_dim=embed_dim,
            dec_dim=dec_dim,
            attn_dim=attn_dim,
        ).to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.extractor = ResNet50SpatialFeatureExtractor(device=self.device)

    @torch.no_grad()
    def generate(self, image: Image.Image, max_len: int | None = None) -> str:
        max_len = max_len or self.vocab.max_len

        fmap = self.extractor.extract(image).unsqueeze(0).to(self.device)  # (1,2048,7,7)
        enc, h, c = self.model.encode_image(fmap)

        cur = torch.tensor([self.vocab.start_idx], dtype=torch.long, device=self.device)
        out_tokens = []

        for _ in range(max_len - 1):
            emb = self.model.embedding(cur)          # (1,embed)
            context, _ = self.model.attn(enc, h)     # (1,dec_dim)

            x = torch.cat([emb, context], dim=-1)    # (1, embed+dec_dim)
            h, c = self.model.lstm(x, (h, c))
            logits = self.model.fc(h)                # (1,V)
            next_id = int(torch.argmax(logits, dim=-1).item())

            if next_id == self.vocab.end_idx:
                break

            out_tokens.append(self.vocab.idx2word[next_id])
            cur = torch.tensor([next_id], dtype=torch.long, device=self.device)

        return " ".join(out_tokens).strip()
