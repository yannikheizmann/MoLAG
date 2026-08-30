# Installation

MoLAG requires Python 3.11 or newer. The locked environment uses
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/yannikheizmann/MoLAG.git
cd MoLAG
uv sync --frozen
```

The default finetuning profile uses bfloat16 arithmetic and is intended for a
CUDA-capable system. Device and precision settings can be overridden through the CLI
or a copied experiment configuration.

For development and documentation builds, install the optional groups:

```bash
uv sync --frozen --extra dev --extra docs
```
