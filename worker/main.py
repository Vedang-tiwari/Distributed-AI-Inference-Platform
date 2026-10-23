import time
import json
import logging
import signal
import sys
from api.core.config import settings
from core.db.session import SessionLocal, InferenceJob, init_db
from core.cache.redis_client import cache_client
from worker.batcher.queue_manager import queue_manager
from worker.batcher.dynamic_batcher import dynamic_batcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (Worker) %(message)s")
logger = logging.getLogger("worker")

running = True


def signal_handler(sig, frame):
    global running
    logger.info("Termination signal received. Shutting down worker gracefully...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def process_async_job(job_payload: dict):
    job_id = job_payload.get("job_id")
    model_name = job_payload.get("model", "resnet")
    version = job_payload.get("version", "v1")
    input_data = job_payload.get("input")
    parameters = job_payload.get("parameters", {})

    logger.info(f"Processing job {job_id} on model {model_name}:{version}...")
    queue_manager.update_job_status(job_id, "running")

    # Update DB
    db = SessionLocal()
    try:
        db_job = db.query(InferenceJob).filter(InferenceJob.id == job_id).first()
        if db_job:
            db_job.status = "running"
            db.commit()

        # Run model inference
        model = dynamic_batcher.get_model(model_name, version)
        results = model.predict_batch([input_data], parameters)
        prediction = results[0] if results else None

        # Mark completed
        queue_manager.update_job_status(job_id, "completed", result=prediction)
        if db_job:
            db_job.status = "completed"
            db_job.result = json.dumps(prediction)
            db.commit()

        logger.info(f"Job {job_id} successfully completed.")
    except Exception as e:
        logger.error(f"Failed job {job_id}: {e}", exc_info=True)
        queue_manager.update_job_status(job_id, "failed", error=str(e))
        if db_job:
            db_job.status = "failed"
            db_job.error = str(e)
            db.commit()
    finally:
        db.close()


def run_worker_loop():
    logger.info(f"Starting Inference Worker [{settings.WORKER_ID}] on device [{settings.WORKER_DEVICE}]...")
    init_db()

    last_heartbeat = 0
    while running:
        current_time = time.time()
        if current_time - last_heartbeat > 5:
            queue_manager.register_worker_heartbeat(
                worker_id=settings.WORKER_ID,
                device=settings.WORKER_DEVICE,
                loaded_models=dynamic_batcher.get_loaded_models(),
            )
            last_heartbeat = current_time

        # Poll async pending jobs queue
        raw_job = cache_client.rpop("queue:jobs:pending")
        if raw_job:
            try:
                job_data = json.loads(raw_job)
                process_async_job(job_data)
            except Exception as e:
                logger.error(f"Error parsing job payload: {e}")
        else:
            time.sleep(0.1)


if __name__ == "__main__":
    run_worker_loop()
