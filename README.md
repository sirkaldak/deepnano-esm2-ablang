
# deepnano-esm2-ablang

This repository contains code and pre-trained models for predicting nanobody-antigen binding affinity using DeepNano with ESM2 embeddings. It includes both 8M and 650M ESM2 model variants, along with training scripts and evaluation metrics for nanobody-antigen prediction.

## Getting Started

### Requirements
The code was executed under python=3.9 and torch=1.13.1+cu116, we recommend you to use similar package versions.
Install DeepNano:
git clone https://github.com/ddd9898/DeepNano.git
cd DeepNano
pip install -r requirements.txt
### Model Weights

Due to their large size, **model checkpoints and ESM2 weights are NOT included** in this repository.

- **Trained checkpoints (AbLang+ESM2)** are provided here (Google Drive):  
  https://drive.google.com/drive/folders/101wmck34b_u1tWuYMHXq53AzstgsvYVl?usp=drive_link

- **ESM2 backbone** (choose one):  
  1) Use a local ESM2 folder (recommended on servers).  
  2) Download from Hugging Face (facebook/esm). *(Optional; requires internet/cache)*

After downloading:
- Put checkpoints anywhere you like (e.g., `output/checkpoint/`), and pass the file path via `--ckpt`.
- Provide the ESM2 directory path via `--esm2`.
### Training the Model

This repo expects the ESM2 model folder to exist under:

- `DeepNano/models/<ESM2_NAME>`  (recommended), **or**
- `DeepNano/<ESM2_NAME>`

Example folder name: `esm2_t6_8M_UR50D`, `esm2_t33_650M_UR50D`, etc.

Run training:

```bash
# activate your conda env
conda activate deepnano_ablang

# train DeepNano_seq (AbLang for nanobody + ESM2 for antigen)
CUDA_VISIBLE_DEVICES=0 python train_Sabdab.py \
  --Model 0 \
  --finetune 1 \
  --ESM2 esm2_t6_8M_UR50D \
  --epochs 10 \
  --bs 32 \
  --lr 5e-5

### Running Inference (NAI_test.csv / any CSV with sequences)

Use `eval_nai_seq.py` to run inference and (optionally) compute metrics if your CSV contains labels.

**Command:**
```bash
conda activate deepnano_ablang

python eval_nai_seq.py \
  --csv data/Sabdab/NAI_test.csv \
  --ckpt /path/to/your_checkpoint.model \
  --esm2 /path/to/esm2_t6_8M_UR50D \
  --bs 8 \
  --finetune 0 \
  --out_csv NAI_test_predictions.csv


## File Structure

- `README.md`: This file.
- `train_Sabdab.py`: Script to train the model on the Sabdab dataset.
- `models_ablang_esm2.py`: Contains the model architecture for DeepNano with AbLang and ESM2.
- `ablang_encoder.py`: Code for AbLang encoding of nanobody sequences.
- `real_maprot.csv`: Example CSV containing nanobody and antigen sequences for testing.
- `eval_nai_seq.py`: Evaluation script for the NAI sequence dataset.

## Acknowledgements
- The DeepNano model is based on work by [DeepNano authors](https://github.com/DeepNano).
- The ESM2 model is based on work by [Facebook AI](https://huggingface.co/facebook/esm).
