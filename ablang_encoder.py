# models/ablang_encoder.py
from __future__ import annotations
from typing import List, Tuple
import re
import numpy as np
import torch
from torch import nn

# AbLang 允许的字符（报错提示那一串 + mask token）
_ABLANG_ALLOWED = set("MRHKDESTNQCGPAVIFYWL*")

# 常见/更稳的映射
_ABLANG_MAP = {
    "U": "C",  # selenocysteine -> C
    "O": "K",  # pyrrolysine  -> K
    "B": "D",  # D/N -> pick D
    "Z": "E",  # E/Q -> pick E
    "J": "L",  # I/L -> pick L
    "X": "A",  # unknown -> A
}

def sanitize_for_ablang(seq: str) -> Tuple[str, bool]:
    """
    返回: (clean_seq, changed_flag)
    - 统一大写、去空白
    - U/O/B/Z/J/X 用上面映射
    - 其它任何奇怪字符 -> A
    """
    s = re.sub(r"\s+", "", str(seq).upper())
    out = []
    changed = False
    for ch in s:
        if ch in _ABLANG_ALLOWED:
            out.append(ch)
        elif ch in _ABLANG_MAP:
            out.append(_ABLANG_MAP[ch])
            changed = True
        else:
            out.append("A")
            changed = True
    return "".join(out), changed


class AbLangHeavyEncoder(nn.Module):
    """
    AbLang heavy-chain encoder.
    Returns:
      - residue embeddings: (B, Lmax, D_in) as torch.float32
      - mask: (B, Lmax) bool, True for valid residues
    """
    def __init__(self, freeze: bool = True, device: torch.device | str = "cpu"):
        super().__init__()
        import ablang  # pip install ablang

        self.device = torch.device(device)
        self.ab = ablang.pretrained("heavy")
        if freeze:
            self.ab.freeze()

        # 只打印一次日志
        self._sanitize_logged = False
        self._truncate_logged = False

        # 取 AbLang position embedding 上限（你实测是 160）
        self.max_pos = 160
        try:
            m = self.ab.AbRep
            for name, mod in m.named_modules():
                if isinstance(mod, torch.nn.Embedding) and name.endswith("PositionEmbeddings"):
                    self.max_pos = int(mod.num_embeddings)
                    break
        except Exception:
            pass

        # AbLang tokenizer 会额外加 token（< > 等），留点 buffer，避免 index out of range
        self.max_res_len = max(1, self.max_pos - 3)

    @torch.no_grad()
    def forward(self, seqs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        seqs: list of antibody/nanobody sequences (strings)
        """
        # ===== sanitize + truncate before feeding AbLang =====
        seqs_clean = []
        changed_cnt = 0
        trunc_cnt = 0

        for s in seqs:
            cs, changed = sanitize_for_ablang(s)
            if changed:
                changed_cnt += 1

            if len(cs) > self.max_res_len:
                cs = cs[:self.max_res_len]
                trunc_cnt += 1

            seqs_clean.append(cs)

        if (changed_cnt > 0) and (not self._sanitize_logged):
            print(
                f"[AbLang] sanitized {changed_cnt}/{len(seqs)} seqs | "
                f"map: O->K U->C B->D Z->E J->L X->A others->A"
            )
            self._sanitize_logged = True

        if (trunc_cnt > 0) and (not self._truncate_logged):
            print(
                f"[AbLang] truncated {trunc_cnt}/{len(seqs)} seqs to max_res_len={self.max_res_len} "
                f"(max_pos={self.max_pos})"
            )
            self._truncate_logged = True

        seqs = seqs_clean
        # =========================================

        # AbLang returns a list/np array; rescoding is per-residue 768-d
        res = self.ab(seqs, mode="rescoding")  # list of (Li, 768) or np.ndarray

        if isinstance(res, np.ndarray):
            # already padded: (B, L, D)
            x = torch.from_numpy(res).float()
            mask = torch.ones(x.shape[:2], dtype=torch.bool)
            return x.to(self.device), mask.to(self.device)

        # else: list of arrays with variable lengths
        arrs = [np.asarray(a) for a in res]
        lens = [a.shape[0] for a in arrs]
        d = arrs[0].shape[1]
        Lmax = max(lens)
        B = len(arrs)

        x = torch.zeros((B, Lmax, d), dtype=torch.float32)
        mask = torch.zeros((B, Lmax), dtype=torch.bool)
        for i, a in enumerate(arrs):
            li = a.shape[0]
            x[i, :li] = torch.as_tensor(a, dtype=torch.float32)
            mask[i, :li] = True

        return x.to(self.device), mask.to(self.device)
