import csv
import os
import argparse
import inspect
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ✅ 改这里：用 AbLang(Nanobody) + ESM2(Antigen) 的新模型文件
from models.models_ablang_esm2 import DeepNano_seq

from utils.evaluate import evaluate


def strip_prefix(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        for p in ["module.", "model.", "net."]:
            if nk.startswith(p):
                nk = nk[len(p):]
        new_sd[nk] = v
    return new_sd


def resolve_esm2_path(esm2_arg: str) -> str:
    if os.path.isdir(esm2_arg):
        return esm2_arg
    cand = os.path.join("./models", esm2_arg)
    if os.path.isdir(cand):
        return cand
    return esm2_arg


def infer_hidden_size(esm2_path_or_name: str) -> int:
    s = esm2_path_or_name
    hidden_map = {
        "t6_8M": 320,
        "t12_35M": 480,
        "t30_150M": 640,
        "t33_650M": 1280,
        "t36_3B": 2560,
        "t48_15B": 5120,
    }
    for k, v in hidden_map.items():
        if k in s:
            return v
    return 1280


class NAISeqCSVDataset(Dataset):
    """
    兼容 NAI_test.csv
    返回：(id1, seq1, id2, seq2, label)
    """
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        cols = list(self.df.columns)

        def pick(*cands):
            for c in cands:
                if c in cols:
                    return c
            return None

        self.col_id1 = pick("ID_nanobody", "id_nanobody", "nanobody_id", "ID1", "id1", "nbo")
        self.col_id2 = pick("ID_antigen", "id_antigen", "antigen_id", "ID2", "id2", "at0")
        self.col_seq1 = pick("seq_nanobody", "nanobody_seq", "seq1", "Ab", "ab", "antibody")
        self.col_seq2 = pick("seq_antigen", "antigen_seq", "seq2", "Ag", "ag", "antigen")
        self.col_y = pick("Interaction label", "label", "labels", "y", "Y", "Interaction_label")

        missing = []
        for name, col in [
            ("id1", self.col_id1),
            ("id2", self.col_id2),
            ("seq1", self.col_seq1),
            ("seq2", self.col_seq2),
            ("label", self.col_y),
        ]:
            if col is None:
                missing.append(name)

        if missing:
            raise RuntimeError(
                f"[NAISeqCSVDataset] 识别不到必要列：{missing}\n"
                f"你的 CSV 列为：{cols}\n"
                f"需要至少包含：ID/seq/label（例如 ID_nanobody, ID_antigen, seq_nanobody, seq_antigen, Interaction label）"
            )

        self.df[self.col_seq1] = self.df[self.col_seq1].astype(str)
        self.df[self.col_seq2] = self.df[self.col_seq2].astype(str)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        id1 = str(r[self.col_id1])
        id2 = str(r[self.col_id2])
        s1 = str(r[self.col_seq1])
        s2 = str(r[self.col_seq2])
        y = float(r[self.col_y])
        return id1, s1, id2, s2, y


def collate_keep_strings(batch):
    id1 = [b[0] for b in batch]
    s1  = [b[1] for b in batch]
    id2 = [b[2] for b in batch]
    s2  = [b[3] for b in batch]
    y   = torch.tensor([b[4] for b in batch], dtype=torch.float32)
    return id1, s1, id2, s2, y


def forward_call(model, seq1, seq2, device):
    sig = inspect.signature(model.forward)
    params = list(sig.parameters.keys())
    if params and params[0] == "self":
        params = params[1:]
    n_params = len(params)

    if n_params == 2:
        return model(seq1, seq2)
    if n_params == 3:
        return model(seq1, seq2, device)
    if n_params >= 4:
        return model(seq1, seq2, None, device)
    return model(seq1, seq2)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/Sabdab/NAI_test.csv")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--esm2", required=True, help="ESM2 本地目录或名字，例如 ./models/esm2_t33_650M_UR50D 或 esm2_t33_650M_UR50D")
    ap.add_argument("--finetune", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--hidden_size", type=int, default=None, help="可选：手动指定 hidden_size（650M=1280, 150M=640, 35M=480, 8M=320）")
    ap.add_argument("--out_csv", default="NAI_test_predictions.csv", help="保存逐样本预测的CSV路径")
    ap.add_argument("--save_seq", action="store_true", help="可选：把序列也写进CSV（文件会很大）")
    ap.add_argument("--th", type=float, default=0.5, help="把概率转0/1的阈值，用于输出 y_pred")
    args = ap.parse_args()

    # ✅ 小提示：你之前遇到的 “Disabling PyTorch because PyTorch>=2.1...” 就是 transformers 版本太新
    # 解决：pip install transformers==4.30.2 tokenizers==0.13.3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    esm2_path = resolve_esm2_path(args.esm2)
    hidden_size = args.hidden_size if args.hidden_size is not None else infer_hidden_size(esm2_path)

    ds = NAISeqCSVDataset(args.csv)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, collate_fn=collate_keep_strings)

    model = DeepNano_seq(pretrained_model=esm2_path, hidden_size=hidden_size, finetune=args.finetune).to(device)
    model.eval()

    w = torch.load(args.ckpt, map_location=device)
    if isinstance(w, dict) and "state_dict" in w:
        w = w["state_dict"]
    w = strip_prefix(w)

    incompatible = model.load_state_dict(w, strict=False)
    print("load ckpt done.")
    if getattr(incompatible, "missing_keys", None):
        mk = incompatible.missing_keys
        if mk:
            print("missing keys:", mk[:20], "..." if len(mk) > 20 else "")
    if getattr(incompatible, "unexpected_keys", None):
        uk = incompatible.unexpected_keys
        if uk:
            print("unexpected keys:", uk[:20], "..." if len(uk) > 20 else "")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    header = ["ab_id", "ag_id"]
    if args.save_seq:
        header += ["ab_seq", "ag_seq"]
    header += ["y_true", "p_ave", "p_min", "p_max", "p_ens", "y_pred"]

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        GT, Pre = [], []

        for id1, s1, id2, s2, y in dl:
            y = y.to(device)
            out = forward_call(model, s1, s2, device)

            # DeepNano_seq 通常返回 (p_ave, p_min, p_max)；这里做 ensemble
            if isinstance(out, (tuple, list)) and len(out) >= 3 and torch.is_tensor(out[0]):
                p_ave, p_min, p_max = out[0], out[1], out[2]
                pred = (p_ave + p_min + p_max) / 3.0

                p_ave_list = p_ave.view(-1).detach().cpu().tolist()
                p_min_list = p_min.view(-1).detach().cpu().tolist()
                p_max_list = p_max.view(-1).detach().cpu().tolist()
            else:
                pred = out[0] if isinstance(out, (tuple, list)) and len(out) > 0 else out
                p_ave_list = None
                p_min_list = None
                p_max_list = None

            pred_list = pred.view(-1).detach().cpu().tolist() if torch.is_tensor(pred) else list(pred)
            lab_list  = y.view(-1).detach().cpu().tolist()
            y_pred_list = [1 if p >= args.th else 0 for p in pred_list]

            if p_ave_list is None:
                p_ave_list = [None] * len(pred_list)
                p_min_list = [None] * len(pred_list)
                p_max_list = [None] * len(pred_list)

            for a_id, a_seq, g_id, g_seq, yt, pa, pmi, pma, pe, yp in zip(
                id1, s1, id2, s2, lab_list, p_ave_list, p_min_list, p_max_list, pred_list, y_pred_list
            ):
                row = [a_id, g_id]
                if args.save_seq:
                    row += [a_seq, g_seq]
                row += [yt, pa, pmi, pma, pe, yp]
                writer.writerow(row)

            Pre += pred_list
            GT  += lab_list

    prec, rec, acc, f1, top10, top20, top50, auc_roc, auc_pr = evaluate(GT, Pre)

    print(
        f"NAI_test: Top10={top10:.4f}, Top20={top20:.4f}, Top50={top50:.4f}, "
        f"acc={acc:.4f}, recall={rec:.4f}, precision={prec:.4f}, f1={f1:.4f}, "
        f"AUC_ROC={auc_roc:.4f}, AUC_PR={auc_pr:.4f}"
    )
    print(f"saved per-sample predictions to: {args.out_csv}")


if __name__ == "__main__":
    main()
