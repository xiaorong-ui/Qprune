# QPrune

QPrune is a question conditioned visual token pruning method for LLaVA. This repository contains the method implementation and a generic evaluation entry point. It is based on the [CDPruner](https://github.com/Theia-4869/CDPruner) LLaVA codebase (Apache 2.0); QPrune changes are primarily in `llava/model/llava_arch.py`.

## Status

This is a source release draft. The code has been separated from private checkpoints, datasets, experiment outputs, and machine specific launch scripts. Reproduction instructions and benchmark manifests need a final review before publication.

## Setup

Use Python 3.10 and a CUDA environment compatible with PyTorch 2.1.2, then install the package:

```bash
pip install -e .
```

Download a compatible LLaVA checkpoint and the evaluation datasets separately.

## Method configuration

The paper method uses `EC_QFICR_LOCAL_RESIDUAL_MODE=p1_lr_generalized_routed_recovery_candidate`, `EC_QFICR_GENERALIZED_ROUTE_MODE=k128_budget_v1`, and `EC_QFICR_RECOVER_RATIO_CAP=0.20`. Set `--visual_token_num` to the desired token budget. The route policy is independent of K.

## Evaluation entry point

```bash
export EC_QFICR_LOCAL_RESIDUAL_MODE=p1_lr_generalized_routed_recovery_candidate
export EC_QFICR_GENERALIZED_ROUTE_MODE=k128_budget_v1
export EC_QFICR_RECOVER_RATIO_CAP=0.20
python -m llava.eval.model_vqa_loader \
  --model-path /path/to/llava-v1.5-7b \
  --question-file /path/to/questions.jsonl \
  --image-folder /path/to/images \
  --answers-file /path/to/predictions.jsonl \
  --visual_token_num 64 --temperature 0 --conv-mode vicuna_v1
```

Question file format and benchmark scoring follow the upstream LLaVA evaluation code. Check benchmark specific data and scoring protocols before comparing results.

## Attribution

This source is derived from [LLaVA](https://github.com/haotian-liu/LLaVA) and [CDPruner](https://github.com/Theia-4869/CDPruner). Their original licenses and notices apply to the derived files. See `LICENSE`.
