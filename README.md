<h1 align="center">Flow Map Language Models:<br>One-step Language Modeling via Continuous Denoising</h1>

<div align="center">
  
**[Chanhyuk Lee](https://david3684.github.io)**<sup>1</sup>, **[Jaehoon Yoo](https://sites.google.com/view/jaehoon-yoo/홈)**<sup>1</sup>, **[Manan Agarwal](https://mananag007.github.io)**<sup>2</sup>, **[Sheel Shah](https://sheelfshah.github.io)**<sup>2</sup>, **[Jerry Huang](https://jrrhuang.github.io/)**<sup>2</sup>, \
**[Aditi Raghunathan](https://www.cs.cmu.edu/~aditirag/)**<sup>2</sup>, **[Seunghoon Hong](https://maga33.github.io/)**<sup>1</sup>, **[Nicholas M. Boffi](https://nmboffi.github.io/)**<sup>†2</sup>, **[Jinwoo Kim](https://jw9730.github.io/)**<sup>†1</sup>



<sup>1</sup>KAIST &nbsp; <sup>2</sup>Carnegie Mellon University &nbsp; <sup>†</sup>Equal advising
</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2602.16813-b31b1b?style=flat&logo=arxiv)](https://arxiv.org/abs/2602.16813)
[![Project Page](https://img.shields.io/badge/Project_Page-grey?style=flat&logo=github)](https://one-step-lm.github.io/)
[![Blog](https://img.shields.io/badge/Blog-grey?style=flat&logo=rss)](https://one-step-lm.github.io/blog/index.html)
[![Google Drive](https://img.shields.io/badge/Google_Drive-4285F4?style=flat&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1fNAx4LP2RwPBdqDQFQ_gRrYZI9u3Vq15?usp=drive_link)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-FF9D00?style=flat&logoColor=white)](https://huggingface.co/collections/david3684/flm-fmlm)

</div>

## News
- **[2026-06]** Released code for [posterior refinement](https://github.com/MananAg007/posterior-refinement).
- **[2026-06]** Released code for [self-conditioned FMLM](https://github.com/Ugness/self-conditioned-fmlm).
- **[2026-05]** Added huggingface links for the checkpoints. 
- **[2026-04]** We released LM1B/OpenWebText checkpoints for FLM and FMLM. 

## TL;DR

<p align="center">
  <img src="figures/overview.gif" width="100%">
</p>

<p align="center">
  <img src="figures/overview.png" width="100%">
</p>

We introduce **Flow Language Model (FLM)** and its flow-map distilled variant **Flow Map Language Model (FMLM)**, enabling **one-step parallel text generation** through continuous denoising. 

## Overview

**FLM** applies the benefits of continuous image generation to discrete state spaces by encoding text as one-hot vectors and using flow matching to directly map noise to one-hot data. Unlike discrete diffusion, **FLM** gradually denoises all tokens in parallel with a deterministic sample-level ODE, allowing it to represent a superposition of sequences and avoid per-token ancestral sampling — a fundamental bottleneck for discrete diffusion in the few-step regime. We extend this to **FMLM**, where learns the **flow map** which is the direct solution operator of the flow, enabling **one-step parallel language generation**.

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

With the option, we are able to train OpenWebText experiments with 512 batch size on 8 H100 (80GB VRAM), with local batch size of 32.

### Training

Before running, update `data.cache_dir` in the scripts to point to your dataset location. If the directory is empty, the dataset will be automatically downloaded and preprocessed.

Set `algo.teacher_path` to your pre-trained FLM checkpoint before running FMLM distillation.


| Model | Dataset     | Script                                                                     |
| ----- | ----------- | -------------------------------------------------------------------------- |
| FLM   | LM1B        | [scripts/train_lm1b_flm.sh](scripts/train_lm1b_flm.sh)                     |
| FMLM  | LM1B        | [scripts/train_lm1b_fmlm_denoiser.sh](scripts/train_lm1b_fmlm_denoiser.sh) |
| FLM   | OpenWebText | [scripts/train_owt_flm.sh](scripts/train_owt_flm.sh)                       |
| FMLM  | OpenWebText | [scripts/train_owt_fmlm_denoiser.sh](scripts/train_owt_fmlm_denoiser.sh)   |

### Evaluation

Set `CKPT_PATH` in the script to your trained checkpoint before running.


| Model | Dataset     | Script                                                       |
| ----- | ----------- | ------------------------------------------------------------ |
| FLM   | LM1B        | [scripts/gen_ppl_lm1b_flm.sh](scripts/gen_ppl_lm1b_flm.sh)   |
| FMLM  | LM1B        | [scripts/gen_ppl_lm1b_fmlm.sh](scripts/gen_ppl_lm1b_fmlm.sh) |
| FLM   | OpenWebText | [scripts/gen_ppl_owt_flm.sh](scripts/gen_ppl_owt_flm.sh)     |
| FMLM  | OpenWebText | [scripts/gen_ppl_owt_fmlm.sh](scripts/gen_ppl_owt_fmlm.sh)   |

## Checkpoints
### Pretrained Checkpoints

Pretrained FLM and FMLM checkpoints are available at [Google Drive](https://drive.google.com/drive/folders/1fNAx4LP2RwPBdqDQFQ_gRrYZI9u3Vq15?usp=drive_link) or [Huggingface](https://huggingface.co/collections/david3684/flm-fmlm).


| Model | Dataset     | Checkpoint       |
| ----- | ----------- | ---------------- |
| FLM   | LM1B        | `lm1b_flm.ckpt`  |
| FMLM  | LM1B        | `lm1b_fmlm.ckpt` |
| FLM   | OpenWebText | `owt_flm.ckpt`   |
| FMLM  | OpenWebText | `owt_fmlm.ckpt`  |


Set `eval.checkpoint_path` (or `algo.teacher_path` for distillation) to the downloaded checkpoint path when running evaluation or distillation scripts.

### Baseline Checkpoints

Reproduced baseline checkpoints for LM1B are available at [here](https://drive.google.com/drive/folders/1TJO3aFWqI7ukbmjciZ6krAUFlAak1itl?usp=drive_link).

For other checkpoints, mostly for OpenWebText, refer to [Duo](https://github.com/s-sahoo/duo), [SDTT](https://github.com/jdeschena/sdtt), [RDLM](https://github.com/harryjo97/RDLM), [di4c](https://github.com/sony/di4c) repositories.


### Full results 

#### FLM (Undistilled)
<p align="center">
  <img src="figures/flm_figure.png" width="100%">
</p>

<table border="0" style="width: 100%; border-collapse: collapse; border: none;">
  <tr style="border: none;">
    <td style="width: 50%; vertical-align: top; border: none; padding-right: 10px;">
      <h4 align="center">LM1B</h4>
      <table style="margin: 0 auto;">
        <thead>
          <tr>
            <th style="text-align: center;">Step</th>
            <th style="text-align: center;">Gen.PPL</th>
            <th style="text-align: center;">Entropy</th>
          </tr>
        </thead>
        <tbody>
          <tr><td style="text-align: center;"><b>8</b></td><td style="text-align: center;">243.36</td><td style="text-align: center;">2.41</td></tr>
          <tr><td style="text-align: center;"><b>16</b></td><td style="text-align: center;">198.53</td><td style="text-align: center;">4.22</td></tr>
          <tr><td style="text-align: center;"><b>32</b></td><td style="text-align: center;">152.01</td><td style="text-align: center;">4.40</td></tr>
          <tr><td style="text-align: center;"><b>64</b></td><td style="text-align: center;">126.51</td><td style="text-align: center;">4.36</td></tr>
          <tr><td style="text-align: center;"><b>128</b></td><td style="text-align: center;">112.54</td><td style="text-align: center;">4.34</td></tr>
          <tr><td style="text-align: center;"><b>256</b></td><td style="text-align: center;">104.59</td><td style="text-align: center;">4.32</td></tr>
          <tr><td style="text-align: center;"><b>512</b></td><td style="text-align: center;">99.75</td><td style="text-align: center;">4.30</td></tr>
          <tr><td style="text-align: center;"><b>1024</b></td><td style="text-align: center;">96.91</td><td style="text-align: center;">4.29</td></tr>
        </tbody>
      </table>
    </td>
    <td style="width: 50%; vertical-align: top; border: none; padding-left: 10px;">
      <h4 align="center">OpenWebText</h4>
      <table style="margin: 0 auto;">
        <thead>
          <tr>
            <th style="text-align: center;">Step</th>
            <th style="text-align: center;">Gen.PPL</th>
            <th style="text-align: center;">Entropy</th>
          </tr>
        </thead>
        <tbody>
          <tr><td style="text-align: center;"><b>8</b></td><td style="text-align: center;">449.15</td><td style="text-align: center;">5.21</td></tr>
          <tr><td style="text-align: center;"><b>16</b></td><td style="text-align: center;">380.99</td><td style="text-align: center;">5.66</td></tr>
          <tr><td style="text-align: center;"><b>32</b></td><td style="text-align: center;">240.11</td><td style="text-align: center;">5.72</td></tr>
          <tr><td style="text-align: center;"><b>64</b></td><td style="text-align: center;">147.28</td><td style="text-align: center;">5.68</td></tr>
          <tr><td style="text-align: center;"><b>128</b></td><td style="text-align: center;">103.30</td><td style="text-align: center;">5.58</td></tr>
          <tr><td style="text-align: center;"><b>256</b></td><td style="text-align: center;">82.05</td><td style="text-align: center;">5.48</td></tr>
          <tr><td style="text-align: center;"><b>512</b></td><td style="text-align: center;">70.22</td><td style="text-align: center;">5.40</td></tr>
          <tr><td style="text-align: center;"><b>1024</b></td><td style="text-align: center;">62.23</td><td style="text-align: center;">5.33</td></tr>
        </tbody>
      </table>
    </td>
  </tr>
</table>

#### FMLM (Distilled)

<p align="center">
  <img src="figures/fmlm_figure.png" width="100%">
</p>

<table border="0" style="width: 100%; border-collapse: collapse; border: none;">
  <tr style="border: none;">
    <td style="width: 50%; vertical-align: top; border: none; padding-right: 10px;">
      <h4 align="center">LM1B</h4>
      <table style="margin: 0 auto;">
        <thead>
          <tr>
            <th style="text-align: center;">Step</th>
            <th style="text-align: center;">Gen.PPL</th>
            <th style="text-align: center;">Entropy</th>
          </tr>
        </thead>
        <tbody>
          <tr><td style="text-align: center;"><b>1</b></td><td style="text-align: center;">119.34</td><td style="text-align: center;">4.16</td></tr>
          <tr><td style="text-align: center;"><b>2</b></td><td style="text-align: center;">110.19</td><td style="text-align: center;">4.21</td></tr>
          <tr><td style="text-align: center;"><b>4</b></td><td style="text-align: center;">98.76</td><td style="text-align: center;">4.21</td></tr>
          <tr><td style="text-align: center;"><b>8</b></td><td style="text-align: center;">86.32</td><td style="text-align: center;">4.21</td></tr>
          <tr><td style="text-align: center;"><b>16</b></td><td style="text-align: center;">78.35</td><td style="text-align: center;">4.21</td></tr>
          <tr><td style="text-align: center;"><b>32</b></td><td style="text-align: center;">69.21</td><td style="text-align: center;">4.21</td></tr>
        </tbody>
      </table>
    </td>
    <td style="width: 50%; vertical-align: top; border: none; padding-left: 10px;">
      <h4 align="center">OpenWebText</h4>
      <table style="margin: 0 auto;">
        <thead>
          <tr>
            <th style="text-align: center;">Step</th>
            <th style="text-align: center;">Gen.PPL</th>
            <th style="text-align: center;">Entropy</th>
          </tr>
        </thead>
        <tbody>
          <tr><td style="text-align: center;"><b>1</b></td><td style="text-align: center;">168.30</td><td style="text-align: center;">5.17</td></tr>
          <tr><td style="text-align: center;"><b>2</b></td><td style="text-align: center;">133.29</td><td style="text-align: center;">5.25</td></tr>
          <tr><td style="text-align: center;"><b>4</b></td><td style="text-align: center;">111.31</td><td style="text-align: center;">5.26</td></tr>
          <tr><td style="text-align: center;"><b>8</b></td><td style="text-align: center;">86.50</td><td style="text-align: center;">5.36</td></tr>
          <tr><td style="text-align: center;"><b>16</b></td><td style="text-align: center;">63.63</td><td style="text-align: center;">5.29</td></tr>
          <tr><td style="text-align: center;"><b>32</b></td><td style="text-align: center;">45.09</td><td style="text-align: center;">5.25</td></tr>
        </tbody>
      </table>
    </td>
  </tr>
</table>

## BibTeX

```bibtex
@article{lee2026flow,
    title={Flow Map Language Models: One-step Language Modeling via Continuous Denoising},
    author={Chanhyuk Lee and Jaehoon Yoo and Manan Agarwal
            and Sheel Shah and Jerry Huang
            and Aditi Raghunathan and Seunghoon Hong
            and Nicholas M. Boffi and Jinwoo Kim},
    journal={arXiv preprint arXiv:2602.16813},
    year={2026},
}
```

---

## Acknowledgements

This repository is built upon the codebases of **[Duo](https://github.com/s-sahoo/duo)** and **[ReDi](https://github.com/Ugness/ReDi)**.
