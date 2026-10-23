from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseModel(ABC):
    def __init__(self, model_name: str, version: str, device: str = "cpu"):
        self.model_name = model_name
        self.version = version
        self.device = device
        self.is_loaded = False

    @abstractmethod
    def load(self):
        """Load model weights and initialize computational pipeline."""
        pass

    @abstractmethod
    def predict_batch(self, batch: List[Any], parameters: Optional[Dict[str, Any]] = None) -> List[Any]:
        """
        Execute batched forward pass on CPU/GPU.
        Must accept a list of inputs and return a list of outputs corresponding 1:1 with the input batch.
        """
        pass

    def unload(self):
        """Release allocated tensors and memory."""
        self.is_loaded = False

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "device": self.device,
            "is_loaded": self.is_loaded,
        }
