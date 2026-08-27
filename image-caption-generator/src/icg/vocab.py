import json
from dataclasses import dataclass
from typing import Dict, List

@dataclass(frozen=True)
class Vocab:
    word2idx: Dict[str, int]
    idx2word: List[str]
    max_len: int

    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    start_token: str = "<start>"
    end_token: str = "<end>"

    @property
    def pad_idx(self) -> int:
        return self.word2idx[self.pad_token]

    @property
    def start_idx(self) -> int:
        return self.word2idx[self.start_token]

    @property
    def end_idx(self) -> int:
        return self.word2idx[self.end_token]

    @property
    def size(self) -> int:
        return len(self.idx2word)

def load_vocab(path: str) -> Vocab:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return Vocab(word2idx=obj["word2idx"], idx2word=obj["idx2word"], max_len=int(obj["max_len"]))
