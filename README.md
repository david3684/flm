# Continuous Denoising Enables One-step Language Modeling

Official code for the paper **"One-step Language Modeling via Continuous Denoising"**

**[Chanhyuk Lee](https://david3684.github.io)**<sup>1</sup>, **[Jaehoon Yoo](https://sites.google.com/view/jaehoon-yoo/홈)**<sup>1</sup>, **[Manan Agarwal](https://mananag007.github.io)**<sup>2</sup>, **[Sheel Shah](https://sheelfshah.github.io)**<sup>2</sup>, **[Jerry Huang](https://jrrhuang.github.io/)**<sup>2</sup>, **[Aditi Raghunathan](https://www.cs.cmu.edu/~aditirag/)**<sup>2</sup>, **[Seunghoon Hong](https://maga33.github.io/)**<sup>1</sup>, **[Nicholas M. Boffi](https://nmboffi.github.io/)**<sup>†2</sup>, **[Jinwoo Kim](https://jw9730.github.io/)**<sup>†1</sup>

<sup>1</sup>KAIST &nbsp; <sup>2</sup>Carnegie Mellon University &nbsp; <sup>†</sup>Equal advising

[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-lightgrey)](https://github.com/david3684/flm)
[![arXiv](https://img.shields.io/badge/arXiv-Coming%20Soon-lightgrey)](https://github.com/david3684/flm)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://one-step-lm.github.io/)
[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/david3684/flm)

---

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



## Installation & Usage

Coming soon.



## BibTeX

```bibtex
@article{lee2025flm,
  author    = {Lee, Chanhyuk and Yoo, Jaehoon and Agarwal, Manan and Shah, Sheel and Huang, Jerry and Raghunathan, Aditi and Hong, Seunghoon and Boffi, Nicholas M. and Kim, Jinwoo},
  title     = {Continuous Denoising Enables One-step Language Modeling},
  journal   = {arXiv preprint},
  year      = {2025},
}
```

---

## License

See [LICENSE](LICENSE).

## Acknowledgements

This codebase builds upon [DUO](https://github.com/s-sahoo/duo).
