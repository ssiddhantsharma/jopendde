# OpenDDE-Preview


![OpenDDE banner](assets/OpenDDE.png)

![Status](https://img.shields.io/badge/status-preview-orange)
![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

OpenDDE is an open-source, all-atom biomolecular foundation model that turns co-folding into a scalable engine for structure prediction, design, and optimization in drug discovery.

> [!IMPORTANT]
> OpenDDE is a preview release. CLI flags, input/output JSON fields, and released
> checkpoints may change between versions, and predictions are not guaranteed to
> be reproducible across releases. It is not yet intended for production
> pipelines. Please open an issue for bugs, regressions, or feature requests.

![OpenDDE banner](assets/scaling_law.png)
![results](assets/results.png)

## News

- **2026-07-03: OpenDDE-Preview has been released! Read the [technical report](assets/OpenDDE_Technical_Reports.pdf) and visit the [project website](https://aurekaresearch.github.io/OpenDDE-Website).**
    - Model weights can be downloaded from Hugging Face: [opendde.pt](https://huggingface.co/aurekaresearch/OpenDDE/resolve/main/opendde.pt) | [opendde_abag.pt](https://huggingface.co/aurekaresearch/OpenDDE/resolve/main/opendde_abag.pt)
    - The Docker image can be pulled with `docker pull aurekaresearch/opendde:v1`
    - The 2026ARK-AB Benchmark is now available

## Installation

OpenDDE requires Python `>=3.11`. The package pins PyTorch, so install with `uv`
and choose the matching PyTorch backend explicitly.

GPU/CUDA install, for Linux CUDA environments:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install --torch-backend cu126 'opendde[gpu]'
opendde doctor
```

CPU-only install:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install --torch-backend cpu 'opendde[cpu]'
opendde doctor
```

From a source checkout, install editable mode instead:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install --torch-backend cpu -e '.[cpu]'
uv pip install --group dev
opendde doctor
```

For full source and Docker setup notes, see
[docs/inference_instructions.md](docs/inference_instructions.md) and
[docs/docker_installation.md](docs/docker_installation.md). Pull the prebuilt
Docker image with:

```bash
docker pull aurekaresearch/opendde:v1
```

## Model and Runtime Data

OpenDDE reads checkpoints and runtime assets from `OPENDDE_ROOT_DIR`, defaulting
to `~/.cache/opendde` when the environment variable is unset:

```text
$OPENDDE_ROOT_DIR/
├── checkpoint/opendde.pt
├── common/
└── search_database/        # needed for local template/RNA-MSA preprocessing
```

From a source checkout, use the repository helper script to prepare the runtime
layout:

```bash
export OPENDDE_ROOT_DIR=/path/to/opendde_data
bash scripts/download_opendde_data.sh
```

The helper script lives in the source tree and is not installed with the
`opendde` Python package. If you installed OpenDDE from a wheel or package
index, either run the script from a cloned checkout, or let `opendde pred`
download the default checkpoint and common runtime files when they are missing.
You can also place checkpoint files manually under
`$OPENDDE_ROOT_DIR/checkpoint/`.

For a prediction that disables protein MSA, template search, and RNA MSA, the
large `search_database/` files are not needed. From a source checkout, use:

```bash
bash scripts/download_opendde_data.sh --skip-search-database
```

Released checkpoints:

| Checkpoint        | Use case                              | Download                                                                                      |
| ----------------- | ------------------------------------- | --------------------------------------------------------------------------------------------- |
| `opendde.pt`      | General-purpose checkpoint.           | [opendde.pt](https://huggingface.co/aurekaresearch/OpenDDE/resolve/main/opendde.pt)           |
| `opendde_abag.pt` | Checkpoint tuned on antibody-antigen. | [opendde_abag.pt](https://huggingface.co/aurekaresearch/OpenDDE/resolve/main/opendde_abag.pt) |

Use `opendde.pt` with `-n opendde_v1` for the default model. For ABAG runs,
keep the filename as `opendde_abag.pt` and pass it explicitly:

```bash
opendde pred \
  -i input.json \
  -o ./output \
  --load_checkpoint_path "$OPENDDE_ROOT_DIR/checkpoint/opendde_abag.pt"
```

Detailed asset setup, mirrors, and Docker data mounts are documented in
[docs/inference_instructions.md](docs/inference_instructions.md).

## Running Your First Prediction

Save this minimal OpenDDE input as `tiny.json`:

```json
[
    {
        "name": "tiny",
        "modelSeeds": [101],
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "ACDEFGHIK",
                    "count": 1
                }
            }
        ]
    }
]
```

Run a small compatibility-oriented prediction. This disables external feature
searches, so only the checkpoint and common runtime files are required:

```bash
opendde pred \
  -i tiny.json \
  -o ./output \
  -n opendde_v1 \
  --use_msa false \
  --use_template false \
  --use_rna_msa false \
  --sample 1 \
  --step 200 \
  --cycle 10
```

Defaults are applied automatically: inference runs in `fp32`, triangle kernels
use `auto` dispatch, and seeds come from the job's `modelSeeds` unless `--seeds`
is provided. On CPU this example may be slow, but it avoids GPU-only kernels and
large search databases.

Outputs are written under:

```text
output/<job_name>/seed_<seed>/predictions/
```

For production runs, enable the preprocessing features you need, for example
`--use_msa true`, `--use_template true`, or `--use_rna_msa true`. Those paths may
require network access, HMMER/Kalign binaries, and large local search databases;
see the inference guide for details.

## 4-GPU Fold-CP Inference

OpenDDE supports a four-GPU Fold-CP inference mode for larger inputs. Launch it
with `torchrun` so that one process runs on each GPU:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node 4 \
  -m runner.batch_inference pred \
  -i examples/protein_200.json \
  -o ./output_cp4 \
  -n opendde_v1 \
  --use_msa false \
  --use_template false \
  --use_rna_msa false \
  --sample 1 \
  --step 200 \
  --cycle 10 \
  --foldcp_mode distributed \
  --foldcp_size_dp 1 \
  --foldcp_size_cp 4
```

`--foldcp_size_cp 4` uses a `2 x 2` context-parallel mesh. For normal single-GPU
or CPU inference, omit the Fold-CP flags or use `--foldcp_mode single`.

## Input JSON

OpenDDE input is a top-level list of jobs. Each job contains `sequences` entries
such as `proteinChain`, `dnaSequence`, `rnaSequence`, `ligand`, or `ion`.
`covalent_bonds` is optional and should be added only when explicit covalent
links are needed.

See [docs/infer_json_format.md](docs/infer_json_format.md) for the full schema,
including covalent bonds, ligands, modifications, MSA paths, and template paths.

## CLI Overview

```bash
opendde pred    # run inference
opendde doctor  # inspect Python/CUDA/kernel setup
opendde json    # convert PDB/CIF structures to OpenDDE JSON
opendde msa     # protein MSA preprocessing
opendde mt      # protein MSA + template preprocessing
opendde prep    # protein MSA + template + RNA MSA preprocessing
```

Use `opendde <command> --help` for command-specific options. Public model names
currently include `opendde_v1`; use `--load_checkpoint_path` for alternate
checkpoint files such as `opendde_abag.pt`.

## Documentation

- [Inference instructions](docs/inference_instructions.md)
- [Docker installation](docs/docker_installation.md)
- [Input JSON format](docs/infer_json_format.md)
- [MSA/template/RNA-MSA pipeline](docs/msa_template_pipeline.md)
- [Kernel options](docs/kernels.md)
- [Supported models](docs/supported_models.md)
- [Tutorial](docs/tutorial.md)

## Citation and Acknowledgements

If you use OpenDDE in your work, please cite this software and the related work.
OpenDDE builds on ideas and components from the AlphaFold 3 ecosystem, including
AlphaFold 3, Protenix, OpenFold, and ColabFold.

## License

OpenDDE is released under the Apache-2.0 license. See [LICENSE](LICENSE).

## Hiring

![Hiring](assets/hiring.png)
