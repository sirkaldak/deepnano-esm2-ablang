import os
import sys
import argparse
import logging
import random
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

warnings.filterwarnings("ignore")

# =========================
# ✅ 关键：把 DeepNano 加进 import 路径
# =========================
ROOT = os.path.dirname(os.path.abspath(__file__))              # ~/projects/deepnano_ablang
DEEP = os.path.join(ROOT, "DeepNano")                          # ~/projects/deepnano_ablang/DeepNano
sys.path.insert(0, DEEP)

# ✅ 用 AbLang+ESM2 的模型
from models.models_ablang_esm2 import DeepNano_seq, DeepNano
from utils.dataloader import seqData_Sabdab, seqData_NBAT_Test
from utils.evaluate import evaluate


def get_args():
    parser = argparse.ArgumentParser(
        description='Train AbLang+ESM2 model on Sabdab (NAI) and eval on NBAT test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--Model', type=int, default=0, help='0: DeepNano_seq, 1: DeepNano')
    parser.add_argument('--finetune', type=int, default=1, help='ESM2 finetune mode')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='ckpt name under deepnano_ablang/output/checkpoint/ (must match this architecture)')
    parser.add_argument('--ESM2', type=str, required=True,
                        help='esm2 folder name (e.g., esm2_t33_650M_UR50D)')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--log_interval', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=1998)
    return parser.parse_args()


def set_seed(seed=1998):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def esm2_dir_from_name(name: str) -> str:
    """
    自动探测 ESM2 权重位置：
      1) DeepNano/models/<name>
      2) DeepNano/<name>
    """
    cand1 = os.path.join(DEEP, "models", name)
    cand2 = os.path.join(DEEP, name)
    if os.path.isdir(cand1):
        return cand1
    if os.path.isdir(cand2):
        return cand2
    raise FileNotFoundError(f"Cannot find ESM2 dir: {cand1} OR {cand2}")


def train_one_epoch(model, device, train_loader, optimizer, epoch, BATCH_SIZE, LOG_INTERVAL):
    logging.info(f'Training on {len(train_loader.dataset)} samples...')
    model.train()

    train_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        seqs_nanobody = data[0]
        seqs_antigen  = data[1]
        gt = data[2].float().to(device)

        optimizer.zero_grad()
        p_ave, p_min, p_max = model(seqs_nanobody, seqs_antigen, device)

        loss1 = F.binary_cross_entropy(p_ave.squeeze(), gt)
        loss2 = F.binary_cross_entropy(p_min.squeeze(), gt)
        loss3 = F.binary_cross_entropy(p_max.squeeze(), gt)
        loss = (loss1 + loss2 + loss3) / 3.0

        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            logging.info(
                f"Train epoch: {epoch} [{batch_idx * BATCH_SIZE}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / max(1, len(train_loader)):.0f}%)] Loss: {loss.item():.6f}"
            )

    return train_loss / max(1, len(train_loader))


@torch.no_grad()
def predicting(model, device, loader):
    model.eval()
    total_preds_ave = torch.Tensor()
    total_preds_min = torch.Tensor()
    total_preds_max = torch.Tensor()
    total_labels = torch.Tensor()

    logging.info(f'Make prediction for {len(loader.dataset)} samples...')
    for data in loader:
        seqs_nanobody = data[0]
        seqs_antigen  = data[1]
        g = data[2]

        p_ave, p_min, p_max = model(seqs_nanobody, seqs_antigen, device)

        total_preds_ave = torch.cat((total_preds_ave, p_ave.cpu()), 0)
        total_preds_min = torch.cat((total_preds_min, p_min.cpu()), 0)
        total_preds_max = torch.cat((total_preds_max, p_max.cpu()), 0)
        total_labels    = torch.cat((total_labels, g), 0)

    return (
        total_labels.numpy().flatten(),
        total_preds_ave.numpy().flatten(),
        total_preds_min.numpy().flatten(),
        total_preds_max.numpy().flatten()
    )


if __name__ == '__main__':
    args = get_args()
    set_seed(args.seed)

    # ===== Train setting =====
    BATCH_SIZE   = args.bs
    LR           = args.lr
    LOG_INTERVAL = args.log_interval
    NUM_EPOCHS   = args.epochs

    # ===== Output dirs (固定在 deepnano_ablang/output/) =====
    OUT_LOG  = os.path.join(ROOT, "output", "log")
    OUT_CKPT = os.path.join(ROOT, "output", "checkpoint")
    os.makedirs(OUT_LOG, exist_ok=True)
    os.makedirs(OUT_CKPT, exist_ok=True)

    # ===== model name (keep clean; add "ablang" via add_name only) =====
    model_name = 'DeepNano_seq' if args.Model == 0 else 'DeepNano'

    # ===== logging (强制：文件+控制台都 INFO，且不重复 handler) =====
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # ===== add_name: ALWAYS contains "_ablang_" exactly once =====
    tf_flag = 1 if (args.pretrained is not None) else 0
    add_name = f"_ablang_({args.ESM2})_SabdabData_finetune{args.finetune}_TF{tf_flag}"
    logfile = os.path.join(OUT_LOG, f'log_{model_name}{add_name}.txt')

    fh = logging.FileHandler(logfile, mode='a')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # ===== data =====
    train_csv = os.path.join(DEEP, "data", "Sabdab", "NAI_train.csv")
    val_csv   = os.path.join(DEEP, "data", "Sabdab", "NAI_val.csv")
    test_seqs = os.path.join(DEEP, "data", "Nanobody_Antigen-main", "all_pair_data.seqs.fasta")
    test_pair = os.path.join(DEEP, "data", "Nanobody_Antigen-main", "all_pair_data.pair.tsv")

    trainDataset = seqData_Sabdab(train_csv)
    valDataset   = seqData_Sabdab(val_csv)
    testDataset  = seqData_NBAT_Test(seq_path=test_seqs, pair_path=test_pair)

    train_loader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True, drop_last=True)
    val_loader   = DataLoader(valDataset,   batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    test_loader  = DataLoader(testDataset,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # ===== device =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== ESM2 dir =====
    esm2_dir = esm2_dir_from_name(args.ESM2)

    # ===== hidden map =====
    hidden_map = {
        "esm2_t6_8M_UR50D": 320,
        "esm2_t12_35M_UR50D": 480,
        "esm2_t30_150M_UR50D": 640,
        "esm2_t33_650M_UR50D": 1280,
        "esm2_t36_3B_UR50D": 2560,
        "esm2_t48_15B_UR50D": 5120,
    }
    if args.ESM2 not in hidden_map:
        raise ValueError(f"Unknown ESM2: {args.ESM2}")
    hidden_size = hidden_map[args.ESM2]

    # ===== build model =====
    if args.Model == 0:
        model = DeepNano_seq(
            pretrained_model=esm2_dir,
            hidden_size=hidden_size,
            finetune=args.finetune
        ).to(device)
    else:
        bsite_ckpt = os.path.join(OUT_CKPT, f"DeepNano_site_ablang({args.ESM2})_SabdabData_finetune1_TF0_best.model")
        model = DeepNano(
            pretrained_model=esm2_dir,
            hidden_size=hidden_size,
            finetune=args.finetune,
            Model_BSite_path=bsite_ckpt
        ).to(device)

    # ===== optionally load pretrained (必须同架构) =====
    if args.pretrained is not None:
        ckpt_path = os.path.join(OUT_CKPT, args.pretrained)
        weights = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(weights, strict=True)
        logging.info(f"Loaded pretrained ckpt: {ckpt_path}")

    # ===== optimizer =====
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-4
    )

    logging.info(f'''Starting training (AbLang+ESM2):
ROOT:           {ROOT}
Model_name:     {model_name}
ESM2:           {args.ESM2}
ESM2_dir:       {esm2_dir}
Finetune:       {args.finetune}
Epochs:         {NUM_EPOCHS}
Batch size:     {BATCH_SIZE}
Learning rate:  {LR}
Seed:           {args.seed}
Training size:  {len(trainDataset)}
Val size:       {len(valDataset)}
Test size:      {len(testDataset)}
Device:         {device}
Logfile:        {logfile}
''')

    best_AUC_PR = -1.0
    best_epoch = -1
    best_ckpt_path = None

    # 记录 best 时对应的 test 指标（防止未初始化）
    Top10_test = Top20_test = Top50_test = accuracy_test = recall_test = precision_test = F1_score_test = AUC_ROC_test = AUC_PR_test = 0.0

    model_file_prefix = os.path.join(OUT_CKPT, model_name + add_name)

    for epoch in range(NUM_EPOCHS):
        _ = train_one_epoch(model, device, train_loader, optimizer, epoch, BATCH_SIZE, LOG_INTERVAL)

        # ===== val =====
        g, p_ave, p_min, p_max = predicting(model, device, val_loader)
        p = (p_ave + p_min + p_max) / 3.0
        precision, recall, accuracy, F1_score, Top10, Top20, Top50, AUC_ROC, AUC_PR = evaluate(g, p)

        logging.info(
            "Val: epoch {}: Top10={:.4f},Top20={:.4f},Top50={:.4f},acc={:.4f},recall={:.4f},precision={:.4f},f1={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f}".format(
                epoch, Top10, Top20, Top50, accuracy, recall, precision, F1_score, AUC_ROC, AUC_PR
            )
        )

        # ===== save best by val AUC_PR =====
        if AUC_PR > best_AUC_PR:
            best_AUC_PR = AUC_PR
            best_epoch = epoch
            best_ckpt_path = model_file_prefix + "_best.model"
            torch.save(model.state_dict(), best_ckpt_path)

            logging.info(f"[SAVE] New best @ epoch={epoch}, best_AUC_PR={best_AUC_PR:.4f}")
            logging.info(f"[SAVE] Path: {best_ckpt_path}")

            # ===== test once when best updates =====
            g, p_ave, p_min, p_max = predicting(model, device, test_loader)
            p = (p_ave + p_min + p_max) / 3.0
            precision_test, recall_test, accuracy_test, F1_score_test, Top10_test, Top20_test, Top50_test, AUC_ROC_test, AUC_PR_test = evaluate(g, p)

            logging.info(
                "Test(best): epoch {}: Top10={:.4f},Top20={:.4f},Top50={:.4f},acc={:.4f},recall={:.4f},precision={:.4f},f1={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f}".format(
                    epoch, Top10_test, Top20_test, Top50_test, accuracy_test, recall_test, precision_test, F1_score_test, AUC_ROC_test, AUC_PR_test
                )
            )

        logging.info(f"[BEST] epoch={best_epoch}, best_val_AUC_PR={best_AUC_PR:.4f} | ckpt={best_ckpt_path}")

    logging.info("===== TRAINING DONE =====")
    logging.info(f"Best epoch: {best_epoch}")
    logging.info(f"Best val AUC_PR: {best_AUC_PR:.4f}")
    logging.info(f"Best ckpt path: {best_ckpt_path}")
    logging.info(
        "Best test metrics (from last best update): Top10={:.4f},Top20={:.4f},Top50={:.4f},acc={:.4f},recall={:.4f},precision={:.4f},f1={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f}".format(
            Top10_test, Top20_test, Top50_test, accuracy_test, recall_test, precision_test, F1_score_test, AUC_ROC_test, AUC_PR_test
        )
    )
