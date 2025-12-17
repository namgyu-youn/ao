# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch
import transformers
from huggingface_hub import ModelCard, get_token, whoami
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig

_transformers_version = str(transformers.__version__)
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    PerRow,
)
from torchao.quantization.quantize_.workflows import (
    Int4ChooseQParamsAlgorithm,
    Int4PackingFormat,
)

safe_serialization = _transformers_version >= "5"


def _get_username():
    token = get_token()
    username = whoami(token=token)["name"]
    return username


MODEL_CARD = """---
base_model: {base_model}
tags:
- transformers
- torchao
- {model_type}
license: apache-2.0
language:
- en
---

# {quant} {base_model} model

- **Developed by:** {username}
- **License:** apache-2.0
- **Quantized from Model:** {base_model}
- **Quantization Method:** {quant}

# Model Performance

## Perplexity (lm-eval)

| Benchmark |                |                      |
|-----------|----------------|----------------------|
|           | {base_model}   | {quantized_model}    |
| mmlu_pro  | To be filled   | To be filled         |

<details>
<summary>Reproduce Perplexity Results</summary>

```Shell
# Baseline
lm_eval --model hf --model_args pretrained={base_model} --tasks mmlu_pro --device cuda:0 --batch_size 1 --limit 100

# Quantized model
lm_eval --model hf --model_args pretrained={quantized_model} --tasks mmlu_pro --device cuda:0 --batch_size 1 --limit 100
```
</details>

## Throughput & Latency (vLLM)

| Benchmark           |                |                      |
|---------------------|----------------|----------------------|
|                     | {base_model}   | {quantized_model}    |
| Throughput (tok/s)  | To be filled   | To be filled         |
| Latency (ms)        | To be filled   | To be filled         |

<details>
<summary>Reproduce Throughput & Latency Results</summary>

```Shell
# Baseline
vllm bench throughput --model {base_model} --input-len 1 --output-len 512 --num-prompts 100

# Quantized model
vllm bench throughput --model {quantized_model} --input-len 1 --output-len 512  --num-prompts 100
```
</details>

# Resources
- **TorchAO GitHub:** [https://github.com/pytorch/ao](https://github.com/pytorch/ao)
- **TorchAO Documentation:** [https://docs.pytorch.org/ao/stable/index.html](https://docs.pytorch.org/ao/stable/index.html)
"""


_int4_quant_code = """
from torchao.quantization import Int4WeightOnlyConfig
quant_config = Int4WeightOnlyConfig(group_size=128, int4_packing_format="tile_packed_to_4d", int4_choose_qparams_algorithm="hqq")
quantization_config = TorchAoConfig(quant_type=quant_config)
quantized_model = AutoModelForCausalLM.from_pretrained(model_to_quantize, device_map="cuda:0", torch_dtype=torch.bfloat16, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)
"""

_fp8_quant_code = """
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow
quant_config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
quantization_config = TorchAoConfig(quant_type=quant_config)
quantized_model = AutoModelForCausalLM.from_pretrained(model_to_quantize, device_map="cuda:0", torch_dtype=torch.bfloat16, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)
"""

_int8_quant_code = """
from torchao.quantization import Int8WeightOnlyConfig
quant_config = Int8WeightOnlyConfig()
quantization_config = TorchAoConfig(quant_type=quant_config)
quantized_model = AutoModelForCausalLM.from_pretrained(model_to_quantize, device_map="cuda:0", torch_dtype=torch.bfloat16, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)
"""

_int8_int8_quant_code = """
from torchao.quantization import Int8DynamicActivationInt8WeightConfig
quant_config = Int8DynamicActivationInt8WeightConfig()
quantization_config = TorchAoConfig(quant_type=quant_config)
quantized_model = AutoModelForCausalLM.from_pretrained(model_to_quantize, device_map="cuda:0", torch_dtype=torch.bfloat16, quantization_config=quantization_config)
tokenizer = AutoTokenizer.from_pretrained(model_id)
"""


def quantize_and_upload(
    model_id: str,
    quant: str,
):
    quant_to_config = {
        "W8A8-FP": Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
        "W4A16-INT": Int4WeightOnlyConfig(
            group_size=128,
            int4_packing_format=Int4PackingFormat.PLAIN,
            int4_choose_qparams_algorithm=Int4ChooseQParamsAlgorithm.TINYGEMM,
        ),
        "W8A16-INT": Int8WeightOnlyConfig(),
        "W8A8-INT": Int8DynamicActivationInt8WeightConfig(),
    }

    quant_to_quant_code = {
        "W8A8-FP": _fp8_quant_code,
        "W8A8-INT": _int8_int8_quant_code,
        "W4A16-INT": _int4_quant_code,
        "W8A16-INT": _int8_quant_code,
    }

    # preparation
    model_to_quantize = model_id

    # quantization
    assert quant in quant_to_config, f"Unsupported quant option: {quant}"
    quant_config = quant_to_config[quant]
    quantization_config = TorchAoConfig(quant_type=quant_config)
    quantized_model = AutoModelForCausalLM.from_pretrained(
        model_to_quantize,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    username = _get_username()

    MODEL_NAME = model_id.split("/")[-1]

    save_to_user_id = username
    save_to = f"{save_to_user_id}/{MODEL_NAME}-{quant}"
    quantized_model_id = save_to
    # model card
    content = MODEL_CARD.format(
        username=username,
        base_model=model_id,
        quantized_model=quantized_model_id,
        model_type=quantized_model.config.model_type,
        quant=quant,
        quant_code=quant_to_quant_code[quant],
        safe_serialization=safe_serialization,
    )
    card = ModelCard(content)

    # Push to hub
    quantized_model.push_to_hub(
        quantized_model_id, safe_serialization=safe_serialization
    )
    tokenizer.push_to_hub(quantized_model_id)
    card.push_to_hub(quantized_model_id)

    # Manual Testing
    prompt = "Hey, are you conscious? Can you talk to me?"
    messages = [
        {
            "role": "system",
            "content": "",
        },
        {"role": "user", "content": prompt},
    ]
    templated_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print("Prompt:", prompt)
    print("Templated prompt:", templated_prompt)
    inputs = tokenizer(
        templated_prompt,
        return_tensors="pt",
    ).to("cuda")
    generated_ids = quantized_model.generate(**inputs, max_new_tokens=128)
    output_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("Response:", output_text[0][len(prompt) :])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model with the specified parameters."
    )
    parser.add_argument(
        "--model_id", type=str, help="Huggingface hub model ID of the model."
    )
    parser.add_argument(
        "--quant",
        type=str,
        help="Quantization method",
    )
    args = parser.parse_args()
    quantize_and_upload(
        args.model_id,
        args.quant,
        args.tasks,
        args.calibration_limit,
        args.max_seq_length,
    )
