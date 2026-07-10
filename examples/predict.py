"""Minimal end-to-end prediction: HF checkpoint -> JAX -> structure.

    uv run --group jax-cuda --group reference python examples/predict.py \
        --seed 0 --sequence MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG

Writes ranked mmCIF(s) + a confidence JSON to the output directory.
"""

from __future__ import annotations

import argparse

import jax

from jopendde.inference import Predictor, predict, spec_from_sequences, summarize

# Ubiquitin (1UBQ), a convenient 76-residue default.
UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True, help="model + sampling seed")
    ap.add_argument("--sequence", default=UBIQUITIN, help="single-chain protein sequence")
    ap.add_argument("--name", default="prediction", help="output name / pdb id")
    ap.add_argument("--out", default="out", help="output directory")
    ap.add_argument("--n-cycle", type=int, default=10, help="trunk recycles")
    ap.add_argument("--n-sample", type=int, default=5, help="diffusion samples")
    ap.add_argument("--n-step", type=int, default=200, help="diffusion steps")
    args = ap.parse_args()

    # 1. Download the torch checkpoint from Hugging Face and convert to JAX.
    #    Needs torch + the vendored OpenDDE (the `reference` group); everything
    #    after this is torch-free.
    predictor = Predictor.from_checkpoint()

    # 2. Featurize the input. `spec_from_sequences` builds the OpenDDE/AF3 input
    #    spec; `featurize` runs OpenDDE's data pipeline into a typed `Features`.
    spec = spec_from_sequences([args.sequence], name=args.name, seed=args.seed)
    inp = predictor.featurize(spec)

    # 3. Predict (pure JAX: model + Features + a random key).
    pred = predict(
        predictor.model,
        inp.feat,
        jax.random.key(args.seed),
        n_cycle=args.n_cycle,
        n_sample=args.n_sample,
        n_step=args.n_step,
    )

    # 4. Per-sample confidence summary (ipTM/pTM/pLDDT/ranking_score/...).
    summary = summarize(inp.feat, pred, predictor.summary_params, n_recycle=args.n_cycle)

    # 5. Write ranked mmCIF(s) + confidence JSON.
    predictor.save(inp, pred, summary, args.out, pdb_id=args.name, seed=args.seed)
    print(f"wrote {args.n_sample} sample(s) to {args.out}/")


if __name__ == "__main__":
    main()
