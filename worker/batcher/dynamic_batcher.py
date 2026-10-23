import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from api.core.config import settings
from worker.models.resnet import ResNetModel
from worker.models.llama import LlamaModel
from worker.models.base import BaseModel

logger = logging.getLogger(__name__)


class BatchItem:
    def __init__(self, request_id: str, input_data: Any, parameters: Dict[str, Any], future: asyncio.Future):
        self.request_id = request_id
        self.input_data = input_data
        self.parameters = parameters
        self.future = future
        self.enqueued_at = time.time()


class DynamicBatcher:
    """
    Advanced Dynamic Request Batcher.
    Collects concurrent inference requests arriving within a configurable time window (max_wait_ms)
    or up to max_batch_size, bundles them into a single batch, and executes a single batched
    forward pass on the underlying model runtime. Demultiplexes outputs back to callers.
    """
    def __init__(self, max_batch_size: int = settings.MAX_BATCH_SIZE, max_wait_ms: int = settings.MAX_WAIT_MS):
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_ms / 1000.0

        # Queues per (model_name, version)
        self._queues: Dict[str, asyncio.Queue] = {}
        self._batch_tasks: Dict[str, asyncio.Task] = {}
        self._models: Dict[str, BaseModel] = {}
        self._lock = None
        self._active_loop = None

        # Telemetry
        self.batches_processed = 0
        self.requests_batched = 0
        self.avg_batch_size = 1.0

        # Preload initial supported models
        self._register_model(ResNetModel(version="v1", device="cpu"))
        self._register_model(ResNetModel(version="v2", device="gpu"))
        self._register_model(LlamaModel(version="v1", device="gpu"))

    def _get_key(self, model: str, version: str) -> str:
        return f"{model}:{version}"

    def _register_model(self, model_instance: BaseModel):
        key = self._get_key(model_instance.model_name, model_instance.version)
        self._models[key] = model_instance

    def get_model(self, model: str, version: str) -> BaseModel:
        key = self._get_key(model, version)
        if key in self._models:
            return self._models[key]

        # Dynamic model instantiation if requested version not pre-instantiated
        if model.lower() == "resnet":
            m = ResNetModel(version=version)
            self._register_model(m)
            return m
        elif model.lower() == "llama":
            m = LlamaModel(version=version)
            self._register_model(m)
            return m
        else:
            # Fallback to generic ResNet structure for testing unknown models
            m = ResNetModel(version=version)
            self._register_model(m)
            return m

    def get_loaded_models(self) -> List[str]:
        return list(self._models.keys())

    def _reset_if_new_loop(self):
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if getattr(self, "_active_loop", None) != current_loop:
            self._active_loop = current_loop
            self._queues = {}
            self._batch_tasks = {}
            self._lock = asyncio.Lock()
        elif getattr(self, "_lock", None) is None:
            self._lock = asyncio.Lock()

    async def _ensure_worker_running(self, key: str):
        self._reset_if_new_loop()
        async with self._lock:
            if key not in self._queues:
                self._queues[key] = asyncio.Queue()
            if key not in self._batch_tasks or self._batch_tasks[key].done():
                self._batch_tasks[key] = asyncio.create_task(self._batching_loop(key))

    async def _batching_loop(self, key: str):
        """
        Background loop accumulating items up to max_batch_size or until max_wait_seconds expires.
        """
        model_name, version = key.split(":")
        queue = self._queues[key]
        model = self.get_model(model_name, version)

        while True:
            try:
                # Wait for the first item
                first_item: BatchItem = await queue.get()
                batch: List[BatchItem] = [first_item]
                start_time = time.time()

                # Collect subsequent items while window has not expired and batch is not full
                while len(batch) < self.max_batch_size:
                    time_left = self.max_wait_seconds - (time.time() - start_time)
                    if time_left <= 0:
                        break
                    try:
                        item: BatchItem = await asyncio.wait_for(queue.get(), timeout=time_left)
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break

                # Execute batch inference
                await self._process_batch(model, batch)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dynamic batching loop for {key}: {e}", exc_info=True)
                await asyncio.sleep(0.01)

    async def _process_batch(self, model: BaseModel, batch: List[BatchItem]):
        batch_size = len(batch)
        self.batches_processed += 1
        self.requests_batched += batch_size
        self.avg_batch_size = round(self.requests_batched / max(self.batches_processed, 1), 2)

        inputs = [item.input_data for item in batch]
        # Use first item's params or merged params
        params = batch[0].parameters

        try:
            # Forward pass run in executor to keep event loop unblocked
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, model.predict_batch, inputs, params)

            # Demultiplex batch output back to individual request futures
            for i, item in enumerate(batch):
                if not item.future.done():
                    res = results[i] if i < len(results) else None
                    item.future.set_result((res, batch_size, model.device))
        except Exception as e:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(e)

    async def submit_request(
        self,
        request_id: str,
        model: str,
        version: str,
        input_data: Any,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, int, str]:
        """
        Submits request to dynamic batcher and returns (prediction_output, batch_size, device).
        """
        key = self._get_key(model, version)
        await self._ensure_worker_running(key)

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        batch_item = BatchItem(
            request_id=request_id,
            input_data=input_data,
            parameters=parameters or {},
            future=future,
        )

        await self._queues[key].put(batch_item)
        return await future


dynamic_batcher = DynamicBatcher()
