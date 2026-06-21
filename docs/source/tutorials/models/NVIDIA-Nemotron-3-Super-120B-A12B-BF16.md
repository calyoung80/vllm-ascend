# NVIDIA-Nemotron-3-Super-120B-A12B-BF16

## Introduction

`NVIDIA-Nemotron-3-Super-120B-A12B-BF16` is a Nemotron-H text generation
model with a hybrid Mamba2, attention, and latent MoE architecture. This
document records the verified `vllm-ascend` deployment path for BF16 inference
on Ascend NPUs, including chunked prefill and `FULL_DECODE_ONLY` graph mode.

The model is currently supported as an experimental model in `vllm-ascend`.

## Supported Features

Refer to [supported models](../../user_guide/support_matrix/supported_models.md)
for the feature matrix.

The verified configuration is:

- BF16 inference
- Tensor parallelism with `tensor_parallel_size=8`
- Chunked prefill with `max_num_batched_tokens=4096`
- Decode graph mode with `cudagraph_mode=FULL_DECODE_ONLY`
- OpenAI-compatible chat completion API

## Environment Preparation

### Model Weight

Download the BF16 model weights from
[Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16)
or prepare an equivalent local model directory.

The model directory should contain the model files plus the model-provided
`chat_template.jinja` and `super_v3_reasoning_parser.py` files.

### Installation

You can use the official `vllm-ascend` docker image:

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
docker run --rm \
  --name vllm-ascend-nemotron \
  --net=host \
  --shm-size=1g \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci4 \
  --device /dev/davinci5 \
  --device /dev/davinci6 \
  --device /dev/davinci7 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /data/models:/data/models \
  -it $IMAGE bash
```

If you do not want to use docker, install `vllm-ascend` from source by following
the [installation guide](../../installation.md).

## Deployment

Set `MODEL` to your local model directory or Hugging Face model identifier. For
offline environments, a local model directory is recommended.

```bash
export MODEL=/data/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

vllm serve "${MODEL}" \
  --served-model-name nemotron-super \
  --trust-remote-code \
  --chat-template "${MODEL}/chat_template.jinja" \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --reasoning-parser super_v3 \
  --reasoning-parser-plugin "${MODEL}/super_v3_reasoning_parser.py" \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 8 \
  --additional-config '{"ascend_compilation_config":{"fuse_norm_quant":false}}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' \
  --host 0.0.0.0 \
  --port 8000
```

### Parameter Notes

- `--chat-template`, `--default-chat-template-kwargs`,
  `--reasoning-parser`, and `--reasoning-parser-plugin` should match the files
  shipped with the model.
- `--max-num-batched-tokens=4096` enables chunked prefill for long and
  concurrent prompts.
- `--compilation-config` enables `FULL_DECODE_ONLY` graph mode for decode.
- `--additional-config '{"ascend_compilation_config":{"fuse_norm_quant":false}}'`
  is part of the verified configuration for this model.

## Functional Verification

For thinking-off chat requests, pass `chat_template_kwargs` explicitly so the
reasoning parser returns the final answer in `message.content`.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron-super",
    "messages": [
      {
        "role": "user",
        "content": "Compute 23 * 19. Reply with only the integer."
      }
    ],
    "max_tokens": 16,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

The expected response content is:

```text
437
```

The following prompts were also verified:

| Prompt | Expected content |
|--------|------------------|
| `Who wrote Pride and Prejudice? Reply with only the author name.` | `Jane Austen` |
| `请只回复这六个字：模型运行正确` | `模型运行正确` |

## Chunked Prefill Verification

Chunked prefill was verified with concurrent long prompts. The tested probe used:

- Concurrency: 2
- Prompt tokens per request: 3313
- Output tokens per request: 9
- `max_num_batched_tokens`: 4096
- Graph mode: `FULL_DECODE_ONLY`

Both requests returned the expected `CHUNKED_PREFILL_OK` response, and the
service log had no Triton-Ascend pointer source error, AICore error, traceback,
or runtime error.

## Performance

The following results were measured with `tensor_parallel_size=8`,
`max_model_len=4096`, `max_num_batched_tokens=4096`, and
`FULL_DECODE_ONLY` graph mode. Each benchmark request used about 3306 prompt
tokens and generated 512 output tokens.

| Concurrency | TTFT avg / max | Prefill tokens/s | Decode tokens/s | E2E output tokens/s |
|-------------|----------------|------------------|-----------------|---------------------|
| 1 | 0.465s / 0.465s | 7110.33 | 48.78 | 42.88 |
| 2 | 0.689s / 0.915s | 7226.58 | 77.21 | 67.58 |
| 4 | 3.697s / 11.730s | 1127.38 | 112.56 | 66.29 |
| 8 | 2.295s / 3.695s | 7158.35 | 153.21 | 130.53 |

Performance depends on the exact hardware, model path, scheduler settings, and
request distribution. Treat these values as a reference for the verified
configuration rather than a guaranteed throughput.
