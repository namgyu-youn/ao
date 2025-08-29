import subprocess
import json
import re

def extract_throughput_metrics(stdout):
   throughput_match = re.search(r'Throughput: ([\d.]+) requests/s, ([\d.]+) total tokens/s', stdout)
   if throughput_match:
       return {
           "req_per_sec": float(throughput_match.group(1)),
           "tok_per_sec": float(throughput_match.group(2))
       }
   return None

def benchmark_model(model_name):
   cmd = [
       "vllm", "bench", "throughput",
       "--model", model_name,
       "--input-len", "512",
       "--output-len", "128",
       "--num-prompts", "100",
       "--dataset-name", "random"
   ]

   try:
       result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
       if result.returncode == 0:
           metrics = extract_throughput_metrics(result.stdout)
           if metrics:
               print(f"Model: {model_name}")
               print(f"Requests/sec: {metrics['req_per_sec']}")
               print(f"Tokens/sec: {metrics['tok_per_sec']}")

               # Save minimal JSON
               with open("benchmark.json", "w") as f:
                   json.dump({"model": model_name, **metrics}, f)
               return metrics
       print("Benchmark failed")
       return None
   except Exception as e:
       print(f"Error: {e}")
       return None

if __name__ == "__main__":
   import sys
   if len(sys.argv) != 2:
       print("Usage: python benchmark.py <model_name>")
       sys.exit(1)
   benchmark_model(sys.argv[1])
