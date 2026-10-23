import time
import math
import hashlib
from typing import Any, Dict, List, Optional
from worker.models.base import BaseModel

# Standard ImageNet candidate classes for demo classification
IMAGENET_CLASSES = [
    "golden_retriever", "tabby_cat", "sports_car", "airliner",
    "espresso", "alp", "monarch_butterfly", "cellular_telephone"
]


class ResNetModel(BaseModel):
    def __init__(self, version: str = "v1", device: str = "cpu"):
        super().__init__(model_name="resnet", version=version, device=device)
        self.num_classes = 1000

    def load(self):
        # Simulated or PyTorch weight loading
        time.sleep(0.01)  # lightweight cold-start simulation
        self.is_loaded = True

    def predict_batch(self, batch: List[Any], parameters: Optional[Dict[str, Any]] = None) -> List[Any]:
        if not self.is_loaded:
            self.load()

        results = []
        for item in batch:
            # Deterministic, realistic feature hash to simulate model forward pass
            item_str = str(item)
            h = int(hashlib.md5(item_str.encode()).hexdigest(), 16)
            class_idx = h % len(IMAGENET_CLASSES)
            predicted_class = IMAGENET_CLASSES[class_idx]

            # Confidence boosted in v2
            base_conf = 0.88 if self.version == "v2" else 0.82
            variance = ((h % 100) / 1000.0)
            confidence = round(min(0.99, base_conf + variance), 4)

            results.append({
                "class_id": class_idx,
                "label": predicted_class,
                "confidence": confidence,
                "top_k": [
                    {"label": predicted_class, "score": confidence},
                    {"label": IMAGENET_CLASSES[(class_idx + 1) % len(IMAGENET_CLASSES)], "score": round((1.0 - confidence) * 0.7, 4)},
                    {"label": IMAGENET_CLASSES[(class_idx + 2) % len(IMAGENET_CLASSES)], "score": round((1.0 - confidence) * 0.3, 4)},
                ]
            })

        return results
