import asyncio
from worker.batcher.dynamic_batcher import DynamicBatcher


def test_dynamic_batcher_single_request():
    async def run():
        batcher = DynamicBatcher(max_batch_size=8, max_wait_ms=30)
        prediction, batch_size, device = await batcher.submit_request(
            request_id="req_single",
            model="resnet",
            version="v1",
            input_data="sample_image_1",
        )
        assert prediction is not None
        assert "label" in prediction
        assert batch_size >= 1
        assert device == "cpu"

    asyncio.run(run())


def test_dynamic_batcher_combines_concurrent_requests():
    """
    Validates that simultaneous requests are grouped together into a single
    batched execution forward pass rather than dispatched one by one.
    """
    async def run():
        batcher = DynamicBatcher(max_batch_size=16, max_wait_ms=50)

        # Launch 6 concurrent inference requests
        tasks = [
            batcher.submit_request(
                request_id=f"req_batch_{i}",
                model="resnet",
                version="v1",
                input_data=f"image_input_matrix_{i}",
            )
            for i in range(6)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 6
        # Each result is a tuple of (prediction, batch_size, device)
        batch_sizes = [res[1] for res in results]

        # Verify that batching occurred (batch_size should be > 1 for grouped requests)
        assert any(b > 1 for b in batch_sizes), f"Expected batching to combine requests, got batch sizes: {batch_sizes}"

        # Verify each caller received valid output
        for pred, b_size, dev in results:
            assert "label" in pred
            assert "confidence" in pred

    asyncio.run(run())


def test_dynamic_batcher_model_version_isolation():
    async def run():
        batcher = DynamicBatcher(max_batch_size=8, max_wait_ms=20)

        res_v1 = await batcher.submit_request(
            request_id="req_v1",
            model="resnet",
            version="v1",
            input_data="input_tensor",
        )
        res_v2 = await batcher.submit_request(
            request_id="req_v2",
            model="resnet",
            version="v2",
            input_data="input_tensor",
        )

        assert res_v1[0]["confidence"] > 0
        assert res_v2[0]["confidence"] > 0

    asyncio.run(run())
