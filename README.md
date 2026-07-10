# jopendde

JAX/Equinox translation of [OpenDDE](https://github.com/aurekaresearch/OpenDDE), an all-atom biomolecular co-folding model. Follows the module-by-module translation approach used in [joltz](https://github.com/nboyd/joltz) (the equivalent translation of Boltz).

`src/jopendde/` is torch-free except for `convert.py`, which converts weights from a PyTorch checkpoint. Inference (`predict`, `summarize`) runs on a converted model and `Features` with only jax/equinox/numpy.

## Install

```bash
uv sync --group jax-cuda    # or: --group jax-cpu
```

Add `--extra cue` for the fused NVIDIA cuEquivariance triangle kernels (CUDA 12).

Weight conversion and featurization additionally need the PyTorch OpenDDE reference, vendored under `./OpenDDE`. Install the `reference` group:

```bash
uv sync --group jax-cuda --group reference
```

torch is pulled as the CPU build (it only extracts weights); JAX owns the GPU. `Predictor.from_checkpoint()` downloads the `opendde_v1` weights on first use.

## Usage

Runnable end-to-end example: [`examples/predict.py`](examples/predict.py).

```bash
uv run --group jax-cuda --group reference python examples/predict.py --seed 0
```

```python
import jax
from jopendde.inference import Predictor, predict, summarize, spec_from_sequences

p = Predictor.from_checkpoint()                          # torch: build + convert
inp = p.featurize(spec_from_sequences(["MQIF..."], name="ubq", seed=0))

# from here, torch-free — needs only (model, Features, SummaryParams):
pred = predict(p.model, inp.feat, jax.random.key(0), n_cycle=10, n_sample=5, n_step=200)
summary = summarize(inp.feat, pred, p.summary_params, n_recycle=10)
p.save(inp, pred, summary, "out/", pdb_id="ubq", seed=0)
```
