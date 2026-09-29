# QPrune

QPrune selects question relevant visual tokens before language model inference.
This repository contains the inference implementation used for the
LLaVA-1.5-7B experiments. It includes the token selector, its LLaVA model
integration, and a JSONL prediction entry point. Checkpoints and datasets are
downloaded separately.

## Installation

Use Python 3.10 and a CUDA environment compatible with PyTorch 2.1.2.

```bash
pip install -e .
```

## Inference

Prepare a merged LLaVA-1.5-7B checkpoint and a JSONL question file. Each
question must have `question_id`, `image`, and `text` fields. Images are
resolved relative to `--image-folder`.

```bash
python -m llava.eval.model_vqa_loader \
  --model-path /path/to/llava-v1.5-7b \
  --question-file /path/to/questions.jsonl \
  --image-folder /path/to/images \
  --answers-file /path/to/answers.jsonl \
  --visual_token_num 64 \
  --temperature 0 \
  --conv-mode vicuna_v1
```

Set `--visual_token_num` to the desired budget per image view. The paper's
LLaVA-1.5 results use 32, 64, and 128. Set `QPRUNE_TRACE_JSONL=1` to save
the selected patch indices alongside the answers. `--resume` resumes a
validated prefix of predictions.

The released selector fixes the paper configuration: 24×24 CLIP patch grid,
CLS attention from layer −2, mixed text/visual probability with the original
agreement gate, density-kernel core selection, entropy-based recovery budget,
and question-dependent recovery routing. See `llava/model/qprune.py` for the
selection implementation. Each sample retains exactly the requested number
of visual tokens. This inference package accepts a merged LLaVA Llama
checkpoint; other backbone families are outside this release.

Benchmark scoring follows the benchmark's official evaluator. The prediction
entry point emits model answers; it does not compute benchmark scores.

## License

The model integration includes code from [LLaVA](https://github.com/haotian-liu/LLaVA).
Its copyright notices are retained in the derived model files. The code is
distributed under Apache 2.0; see [LICENSE](LICENSE).
