#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TorchAoConfig
from torchao.prototype.awq.example import get_calib_dataset
from torchao.prototype.smoothquant import SmoothQuantConfig
from torchao.prototype.smoothquant.core import SmoothQuantStep
from torchao.quantization import quantize_
from torchao.quantization.quant_api import Int8DynamicActivationInt8WeightConfig


def quantize_and_save(model_id: str, output_path: str, alpha: float = 0.5,
                    calibration_limit: int = 10, max_seq_length: int = 512):
   """Quantize model with SmoothQuant and save in HuggingFace format"""

   print(f"Loading model: {model_id}")
   tokenizer = AutoTokenizer.from_pretrained(model_id)
   if tokenizer.pad_token is None:
       tokenizer.pad_token = tokenizer.eos_token

   model = AutoModelForCausalLM.from_pretrained(
       model_id,
       torch_dtype=torch.bfloat16,
       device_map="cpu"
   ).eval()

   # Step 1: Prepare observers
   print("Preparing SmoothQuant observers...")
   quant_config = SmoothQuantConfig(
       base_config=Int8DynamicActivationInt8WeightConfig(),
       step=SmoothQuantStep.PREPARE,
       alpha=alpha,
   )
   quantize_(model, quant_config)

   # Step 2: Calibration
   print("Calibrating...")
   calibration_data = get_calib_dataset(
       tokenizer=tokenizer, n_samples=calibration_limit, block_size=max_seq_length
   )

   with torch.no_grad():
       for i, batch in enumerate(calibration_data):
           print(f"Calibration step {i+1}/{calibration_limit}")
           model(batch)

   # Step 3: Convert to quantized model
   print("Converting to quantized model...")
   quant_config.step = SmoothQuantStep.CONVERT
   quantize_(model, quant_config)

   # Step 4: Prepare for saving
   print("Preparing for saving...")
   quant_config.step = SmoothQuantStep.PREPARE_FOR_LOADING
   model.config.quantization_config = TorchAoConfig(quant_config)

   print(f"Saving to: {output_path}")
   model.save_pretrained(output_path, safe_serialization=False)
   tokenizer.save_pretrained(output_path)
   print("Done!")


def main():
   parser = argparse.ArgumentParser(description="SmoothQuant quantization")
   parser.add_argument("--model", required=True, help="HuggingFace model ID")
   parser.add_argument("--output", required=True, help="Output directory path")
   parser.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant alpha")
   parser.add_argument("--calibration-limit", type=int, default=5, help="Calibration samples")
   parser.add_argument("--max-seq-length", type=int, default=512, help="Max sequence length")

   args = parser.parse_args()

   quantize_and_save(
       model_id=args.model,
       output_path=args.output,
       alpha=args.alpha,
       calibration_limit=args.calibration_limit,
       max_seq_length=args.max_seq_length
   )


if __name__ == "__main__":
   main()
