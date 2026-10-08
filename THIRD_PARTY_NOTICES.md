# Third-party code and licenses

ReSFlow contributions are released under the MIT license in `LICENSE`.
Third-party portions retain their upstream notices and terms; the project
license does not override those terms. Downloaded datasets and model weights
are not bundled and retain their providers' terms.

| Source | Local components | Upstream license copy |
| --- | --- | --- |
| [Statistical Flow Matching](https://github.com/ccr-cheng/statistical-flow-matching) | Base flow models, experiment infrastructure, dataset and evaluation implementations; modified for ReSFlow | [MIT](licenses/SFM.txt) |
| [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) | `models/dit/` | [MIT](licenses/SEDD.txt) |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | Portions of `models/transformer.py` | MIT notice retained in the file |
| [Bayesian Flow Networks](https://github.com/nnaisense/bayesian-flow-networks) | Portions of `models/transformer.py` | [Apache 2.0](licenses/BFN.txt) |
| [NCSNv2](https://github.com/ermongroup/ncsnv2) | `models/cnn/` | [MIT](licenses/NCSNv2.txt) |
| [pytorch_ema](https://github.com/fadel/pytorch_ema) | `models/ema.py` | [MIT](licenses/pytorch_ema.txt) |
| [Selene](https://github.com/FunctionLab/selene) | Genome and genomic signal classes in `datasets/promoter.py` | [Clear BSD](licenses/Selene.txt) |
| [Sei framework](https://github.com/FunctionLab/sei-framework) | `evaluation/sei.py` and externally downloaded Sei model | [Academic and research use only](licenses/Sei.txt) |
| [pytorch-fid](https://github.com/mseitzer/pytorch-fid) / TTUR | `evaluation/fid.py`, `evaluation/inception.py` | [Apache 2.0](licenses/pytorch_fid.txt); existing TTUR copyright retained |
| [GLIDE](https://github.com/openai/glide-text2im) | Timestep embedding adapted in `models/dit/transformer.py` | [MIT](licenses/GLIDE.txt) |

## Sei restriction

The Sei upstream license permits redistribution and use for **academic and
research use only**. ReSFlow's MIT license does not grant broader rights to
Sei code or weights. See the full license in `licenses/Sei.txt`.

## Modifications

The ReSFlow version adds spherical reflow, task configuration profiles,
automatic dataset downloads and preprocessing, checkpoint loading compatibility,
native PyTorch attention support, and evaluation notebooks. Third-party files
may therefore differ from their upstream versions. README figures come from
the ReSFlow manuscript supplied by its authors.
