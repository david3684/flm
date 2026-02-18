# One-step Language Modeling via Continuous Denoising

Official code for the paper **"One-step Language Modeling via Continuous Denoising"**

**[Chanhyuk Lee](https://david3684.github.io)**<sup>1</sup>, **[Jaehoon Yoo](https://sites.google.com/view/jaehoon-yoo/홈)**<sup>1</sup>, **[Manan Agarwal](https://mananag007.github.io)**<sup>2</sup>, **[Sheel Shah](https://sheelfshah.github.io)**<sup>2</sup>, **[Jerry Huang](https://jrrhuang.github.io/)**<sup>2</sup>, **[Aditi Raghunathan](https://www.cs.cmu.edu/~aditirag/)**<sup>2</sup>, **[Seunghoon Hong](https://maga33.github.io/)**<sup>1</sup>, **[Nicholas M. Boffi](https://nmboffi.github.io/)**<sup>†2</sup>, **[Jinwoo Kim](https://jw9730.github.io/)**<sup>†1</sup>

<sup>1</sup>KAIST &nbsp; <sup>2</sup>Carnegie Mellon University &nbsp; <sup>†</sup>Equal advising

[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-lightgrey)](https://github.com/david3684/flm)
[![arXiv](https://img.shields.io/badge/arXiv-Coming%20Soon-lightgrey)](https://github.com/david3684/flm)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://one-step-lm.github.io/)


## TL;DR

We introduce **Flow-based Language Model (FLM)** and its flow-map distilled variant **Flow-map Language Model (FMLM)**, enabling **one-step parallel text generation** through continuous denoising. 



## Overview

<p align="center">
  <img src="figures/overview.gif" width="100%">
</p>

<p align="center">
  <img src="figures/overview.png" width="100%">
</p>

FLM applies the benefits of continuous image generation to discrete state spaces by encoding text as one-hot vectors and using flow matching to directly map noise to one-hot data. Unlike discrete diffusion, FLM **gradually denoises all tokens in parallel**, allowing it to represent a superposition of sequences while capturing correlations between tokens — a fundamental bottleneck for discrete diffusion in the few-step regime.



## How to Run

### Install Dependencies

```bash
pip install torch>=2.3.0
pip install -r requirements.txt
# Install flash-attn separately matching your python / torch version (see https://github.com/Dao-AILab/flash-attention/releases)
pip install flash-attn==2.8.3 --no-build-isolation
```

Our DiT backbone supports `torch.compile` with `max-autotune` for faster training. Enable it by setting the environment variable before running any script:

```bash
export DIT_USE_COMPILE=TRUE
```

With the option, we are able to train OpenWebText experiments with 512 batch size on 8 H100 (80GB VRAM), without gradient accumulation.

### Training

Before running, update `data.cache_dir` in the scripts to point to your dataset location. If the directory is empty, the dataset will be automatically downloaded and preprocessed.

**FLM Training** (1M steps)

| Dataset | Script |
|---|---|
| LM1B | [scripts/train_lm1b_flm.sh](scripts/train_lm1b_flm.sh) |
| OpenWebText | [scripts/train_owt_flm.sh](scripts/train_owt_flm.sh) |

**Flow Map Distillation**

Set `algo.teacher_path` to your pre-trained FLM checkpoint before running.

| Dataset | Script |
|---|---|
| LM1B | [scripts/train_lm1b_flm_distill.sh](scripts/train_lm1b_flm_distill.sh) |
| OpenWebText | [scripts/train_owt_flm_distill.sh](scripts/train_owt_flm_distill.sh) |

**Second Stage Distillation** (optional)

Set `algo.teacher_path_f` to your pre-trained FLM checkpoint and `algo.teacher_path_g` to your distilled backbone from above script.

| Dataset | Script |
|---|---|
| LM1B | [scripts/train_lm1b_flm_distill_second.sh](scripts/train_lm1b_flm_distill_second.sh) |
| OpenWebText | [scripts/train_owt_flm_distill_second.sh](scripts/train_owt_flm_distill_second.sh) |

### Evaluation

Set `CKPT_PATH` in the script to your trained checkpoint before running.

| Model | Dataset | Script |
|---|---|---|
| FLM | LM1B | [scripts/gen_ppl_lm1b_flm.sh](scripts/gen_ppl_lm1b_flm.sh) |
| FLM | OpenWebText | [scripts/gen_ppl_owt_flm.sh](scripts/gen_ppl_owt_flm.sh) |
| FMLM | LM1B | [scripts/gen_ppl_lm1b_flm_distill_double.sh](scripts/gen_ppl_lm1b_flm_distill_double.sh) |
| FMLM | OpenWebText | [scripts/gen_ppl_owt_flm_distill_double.sh](scripts/gen_ppl_owt_flm_distill_double.sh) |



## BibTeX

Coming Soon

---


## Acknowledgements

This codebase builds upon [DUO](https://github.com/s-sahoo/duo).
