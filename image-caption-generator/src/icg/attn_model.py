import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, attn_dim)
        self.W_dec = nn.Linear(dec_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, enc_out: torch.Tensor, h: torch.Tensor):
        """
        enc_out: (B, L, enc_dim)   where L=49
        h:       (B, dec_dim)
        returns: context (B, enc_dim), alpha (B, L)
        """
        score = torch.tanh(self.W_enc(enc_out) + self.W_dec(h).unsqueeze(1))  # (B,L,attn)
        e = self.v(score).squeeze(-1)                                         # (B,L)
        alpha = torch.softmax(e, dim=1)                                       # (B,L)
        context = (enc_out * alpha.unsqueeze(-1)).sum(dim=1)                  # (B,enc_dim)
        return context, alpha


class AttnCaptionModel(nn.Module):
    """
    ResNet50 spatial features (B,2048,7,7) -> (B,49,512)
    Bahdanau attention over 49 locations + LSTMCell decoder
    """
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        enc_dim: int = 2048,
        embed_dim: int = 256,
        dec_dim: int = 512,
        attn_dim: int = 256,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # project encoder channels -> dec_dim
        self.enc_proj = nn.Linear(enc_dim, dec_dim)

        self.attn = BahdanauAttention(enc_dim=dec_dim, dec_dim=dec_dim, attn_dim=attn_dim)
        self.lstm = nn.LSTMCell(input_size=embed_dim + dec_dim, hidden_size=dec_dim)

        self.init_h = nn.Linear(dec_dim, dec_dim)
        self.init_c = nn.Linear(dec_dim, dec_dim)

        self.fc = nn.Linear(dec_dim, vocab_size)

    def encode_image(self, fmap_2048_7_7: torch.Tensor):
        """
        fmap_2048_7_7: (B,2048,7,7)
        returns:
          enc: (B,49,dec_dim)
          h,c: (B,dec_dim)
        """
        B = fmap_2048_7_7.size(0)
        enc = fmap_2048_7_7.permute(0, 2, 3, 1).contiguous().view(B, 49, 2048).float()
        enc = torch.relu(self.enc_proj(enc))   # (B,49,dec_dim)

        enc_mean = enc.mean(dim=1)             # (B,dec_dim)
        h = torch.tanh(self.init_h(enc_mean))
        c = torch.tanh(self.init_c(enc_mean))
        return enc, h, c
