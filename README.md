# Learning Temporal Dynamics for Few-Shot Audio–Visual DeepFake Detection

Official code for the ICISS 2027 paper:

> **Learning Temporal Dynamics for Few-Shot Audio–Visual DeepFake Detection**
> Saptarshi Pani\*, Sajjad Ahmed Shaaz\*, Sergei Romanov, Dmitrii Kaplun, Ram Sarkar
> *International Conference on Information Systems Security (ICISS), 2027*
> \* equal contribution

## Overview

The framework detects audio–visual deepfakes by modelling how a video's audio–visual evidence changes over time:

1. **Segmentation.** Each video is split into non-overlapping 2 s steps. Each step yields one face frame and the matching time slice of the log-Mel spectrogram.
2. **Encoders.** A frozen **CLIP ViT-B/32** encodes each face frame. A small **CNN** encodes each Mel slice.
3. **Coherence Head.** Two 2-layer MLPs project the visual and audio embeddings into a shared space. Their cosine similarity `c_t` gives a per-step audio–visual coherence score. The head is pre-trained with a symmetric InfoNCE loss on **real videos only**.
4. **Value Network.** The fused state `s_t = [v_t ; a_t ; c_t]` passes through a fusion MLP and a causal GRU, which predict a value `V(s_t)` at each step. Positive values indicate fake; negative values indicate real.
5. **Objective.** A TD(λ) loss over shaped per-step rewards is combined with a classification loss on the final-step value: `L = L_TD + w · L_BCE`.
6. **Few-shot adaptation.** The encoders and Coherence Head stay frozen. Only the Value Network is trained, on *k* labelled real and *k* labelled fake videos.

The notebook also includes:
- **Early-exit inference**: the model stops at the first step where `|V_t| > τ`.
- **TD-error localisation plots.**
- **CLIP attention-rollout and Mel-spectrogram visualisations.**

## Repository contents

| File | Description |
|---|---|
| `td-verdict-polyglotfake.ipynb` | End-to-end pipeline on **PolyGlotFake** (binary Real / Fake). It covers data loading, Stage-1 coherence pre-training, Stage-2 TD(λ) value training, evaluation, early exit, TD-error localisation, qualitative figures and *k*-shot value-head adaptation. The saved outputs come from one run on a Kaggle Tesla T4. |
| `requirements.txt` | Python dependencies. |

## Setup

```bash
git clone <this-repo-url>
cd <repo>
pip install -r requirements.txt
```

The code was run with Python 3.12 and PyTorch 2.x on a CUDA GPU. The CPU will work but is slow, because CLIP encodes every frame on every pass.

## Data

Download PolyGlotFake from its original authors (Hou et al., ICPR 2024) under their licence. This repository does not redistribute the dataset.

The notebook expects pre-extracted frames and Mel-spectrogram images in this layout:

```
<DATASET_ROOT>/
├── Real/
│   └── <video_id>/
│       ├── Frames/      # face crops, one PNG per 2 s step (sorted by filename)
│       └── Audio/       # a single *spectrogram*.png (log-Mel, 128 bins, fmax 8 kHz)
└── Fake/
    └── <video_id>/
        ├── Frames/
        └── Audio/
```

Preprocessing works as described in Sec. 2.1 of the paper:
- OpenCV samples one frame every 2 s.
- A Haar-cascade face detector finds faces, and the largest face is cropped and resized to 224×224.
- MoviePy extracts the audio, which is converted to a log-Mel spectrogram.

Set `CFG.DATASET_ROOT` in the config cell to your `<DATASET_ROOT>`.

## Usage

Open the notebook and run the cells in order.

- **Cells 0–15: training and evaluation.** The model trains with 8 epochs of Stage 1 and 20 epochs of Stage 2, then is evaluated on the held-out split. Outputs:
  - `best_tdverdict.pth`
  - `tdverdict_results.json`
  - `early_exit_curve.png`
  - `td_error_localization.png`
- **Cells 16–22: qualitative figures.** They write `qual_real_*.{png,pdf}` and `qual_fake_*.{png,pdf}`.
- **Cells 23+: *k*-shot adaptation.** `few_shot_adapt(model, train_data, test_loader, clip_preprocess, k=K)` re-initialises the Value Network and trains it on *k* real and *k* fake training videos. The frozen CLIP encoder, audio encoder and Coherence Head are reused.

Key hyperparameters are in `CFG`:

| Parameter | Value |
|---|---|
| `GAMMA` | 0.95 |
| `LAMBDA` | 0.8 |
| `ETA` (shaped-reward weight) | 0.10 |
| `MAX_STEPS` | 16 |
| `BATCH_SIZE` | 8 |
| `SEED` | 42 |

<!--
## Citation

```bibtex
@inproceedings{pani2027temporal,
  title     = {Learning Temporal Dynamics for Few-Shot Audio--Visual DeepFake Detection},
  author    = {Pani, Saptarshi and Shaaz, Sajjad Ahmed and Romanov, Sergei and Kaplun, Dmitrii and Sarkar, Ram},
  booktitle = {International Conference on Information Systems Security (ICISS)},
  year      = {2027}
}
```
-->

## Acknowledgements

This work was supported by Grant No. 26-91-20001 from the Russian Science Foundation. The project builds on [OpenAI CLIP](https://github.com/openai/CLIP) and uses the PolyGlotFake and FakeAVCeleb datasets.
