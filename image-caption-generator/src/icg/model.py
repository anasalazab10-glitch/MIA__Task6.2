import torch
import torch.nn as nn

class ImageCaptioningModel(nn.Module):
    def __init__(self, feature_dim: int, embed_size: int, hidden_size: int, vocab_size: int, pad_idx: int):
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, hidden_size)
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, image_features: torch.Tensor, caption_tokens: torch.Tensor) -> torch.Tensor:
        # image_features: (B, 2048)
        # caption_tokens: (B, L)
        img_h = torch.relu(self.feature_proj(image_features))  # (B, hidden)
        h0 = img_h.unsqueeze(0).contiguous()                   # (1, B, hidden)
        c0 = torch.zeros_like(h0)

        x = self.embedding(caption_tokens)                     # (B, L, embed)
        out, _ = self.lstm(x, (h0, c0))                        # (B, L, hidden)
        logits = self.fc(out)                                  # (B, L, V)
        return logits
