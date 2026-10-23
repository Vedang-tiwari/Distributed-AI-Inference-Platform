import time
import hashlib
from typing import Any, Dict, List, Optional
from worker.models.base import BaseModel


class LlamaModel(BaseModel):
    def __init__(self, version: str = "v1", device: str = "gpu"):
        super().__init__(model_name="llama", version=version, device=device)

    def load(self):
        time.sleep(0.02)
        self.is_loaded = True

    def predict_batch(self, batch: List[Any], parameters: Optional[Dict[str, Any]] = None) -> List[Any]:
        if not self.is_loaded:
            self.load()

        params = parameters or {}
        max_tokens = params.get("max_tokens", 64)
        temperature = params.get("temperature", 0.7)

        results = []
        for prompt_input in batch:
            prompt_text = str(prompt_input.get("prompt", prompt_input) if isinstance(prompt_input, dict) else prompt_input)
            prompt_tokens = len(prompt_text.split())

            # Generate synthetic continuation demonstrating autoregressive behavior
            continuation = (
                f"is a distributed concept that powers high-throughput AI services with "
                f"low P99 latency and efficient GPU tensor parallelism."
            )
            generated_text = f"{prompt_text} {continuation}"
            completion_tokens = min(max_tokens, len(continuation.split()))

            results.append({
                "prompt": prompt_text,
                "generated_text": generated_text,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "finish_reason": "stop"
            })

        return results
