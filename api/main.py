import time
import uuid
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.core.config import settings
from core.db.session import init_db
from api.routes import predict, models, jobs, metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (API) %(name)s: %(message)s"
)
logger = logging.getLogger("api-gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Distributed AI Inference Platform...")
    init_db()
    logger.info("Database schemas and seed records verified.")
    yield
    logger.info("Shutting down Distributed AI Inference Platform.")


app = FastAPI(
    title="Distributed AI Inference Platform",
    description=(
        "Production-grade miniature model serving platform featuring dynamic batching, "
        "model versioning, canary deployments, rate limiting, request queuing, and real-time observability."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_and_request_id(request: Request, call_next):
    start_time = time.time()
    req_id = request.headers.get("X-Request-ID", f"req_{uuid.uuid4().hex[:12]}")

    response = await call_next(request)

    process_time = round((time.time() - start_time) * 1000.0, 2)
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Response-Time-Ms"] = str(process_time)
    return response


# Include Routers
app.include_router(predict.router)
app.include_router(models.router)
app.include_router(jobs.router)
app.include_router(metrics.router)


@app.get("/", tags=["Root"])
def root():
    return {
        "service": "Distributed AI Inference Platform",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "models": "/v1/models",
            "predict_resnet": "/v1/models/resnet/predict",
            "generate_llama": "/v1/models/llama/generate",
            "jobs": "/v1/jobs",
            "metrics": "/v1/metrics",
            "health": "/health",
        },
    }
