import datetime
import json
import logging
from typing import Generator
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    Text,
    ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from api.core.config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    email = Column(String(128), unique=True, index=True, nullable=True)
    tier = Column(String(32), default="free")  # 'free', 'premium', 'enterprise'
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(128), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    tier = Column(String(32), default="free")
    is_active = Column(Boolean, default=True)
    rate_limit_rpm = Column(Integer, default=100)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class ModelRegistryEntry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(64), index=True, nullable=False)
    version = Column(String(32), nullable=False)
    framework = Column(String(32), default="pytorch")
    device = Column(String(32), default="cpu")
    artifact_uri = Column(String(256), default="")
    status = Column(String(32), default="READY")  # READY, CANARY, DEPRECATED
    description = Column(String(256), default="")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class CanaryDeployment(Base):
    __tablename__ = "canary_deployments"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(64), unique=True, index=True, nullable=False)
    weights_json = Column(Text, nullable=False, default="{}")
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class InferenceJob(Base):
    __tablename__ = "inference_jobs"

    id = Column(String(64), primary_key=True, index=True)
    user_id = Column(Integer, nullable=True)
    model = Column(String(64), index=True, nullable=False)
    version = Column(String(32), nullable=True)
    input_data = Column(Text, nullable=False)
    parameters = Column(Text, default="{}")
    status = Column(String(32), default="queued", index=True)  # queued, running, completed, failed
    result = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class UsageLog(Base):
    __tablename__ = "usage_logs"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(64), index=True)
    user_id = Column(Integer, nullable=True)
    model = Column(String(64), index=True)
    version = Column(String(32), nullable=True)
    device = Column(String(32), default="cpu")
    worker_id = Column(String(64), nullable=True)
    latency_ms = Column(Float, default=0.0)
    status_code = Column(Integer, default=200)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)


connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Seed default admin user & master API key if not existing
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            admin_user = User(username="admin", email="admin@platform.ai", tier="enterprise")
            db.add(admin_user)
            db.commit()
            db.refresh(admin_user)

        master_key = db.query(APIKey).filter(APIKey.key == settings.DEFAULT_MASTER_KEY).first()
        if not master_key:
            master_key = APIKey(
                key=settings.DEFAULT_MASTER_KEY,
                user_id=admin_user.id,
                tier="enterprise",
                is_active=True,
                rate_limit_rpm=settings.PREMIUM_TIER_RPM,
            )
            db.add(master_key)

        # Also seed a demo free API key
        demo_user = db.query(User).filter(User.username == "demo").first()
        if not demo_user:
            demo_user = User(username="demo", email="demo@platform.ai", tier="free")
            db.add(demo_user)
            db.commit()
            db.refresh(demo_user)

        demo_key = db.query(APIKey).filter(APIKey.key == "ai_live_demo_free_123456").first()
        if not demo_key:
            demo_key = APIKey(
                key="ai_live_demo_free_123456",
                user_id=demo_user.id,
                tier="free",
                is_active=True,
                rate_limit_rpm=settings.FREE_TIER_RPM,
            )
            db.add(demo_key)

        # Seed initial models
        models_to_seed = [
            ("resnet", "v1", "pytorch", "cpu", "s3://models/resnet/v1/model.pt", "ResNet-50 v1 base model"),
            ("resnet", "v2", "pytorch", "gpu", "s3://models/resnet/v2/model.pt", "ResNet-50 v2 optimized model"),
            ("llama", "v1", "pytorch", "gpu", "s3://models/llama/v1/model.pt", "Llama-7B simulated language model"),
        ]
        for name, version, framework, device, uri, desc in models_to_seed:
            entry = db.query(ModelRegistryEntry).filter(
                ModelRegistryEntry.name == name,
                ModelRegistryEntry.version == version
            ).first()
            if not entry:
                db.add(ModelRegistryEntry(
                    name=name,
                    version=version,
                    framework=framework,
                    device=device,
                    artifact_uri=uri,
                    status="READY",
                    description=desc,
                ))

        # Seed canary deployment configuration for resnet
        resnet_canary = db.query(CanaryDeployment).filter(CanaryDeployment.model_name == "resnet").first()
        if not resnet_canary:
            db.add(CanaryDeployment(
                model_name="resnet",
                weights_json=json.dumps({"v1": 0.2, "v2": 0.8})
            ))

        db.commit()
    except Exception as e:
        logger.error(f"Failed to seed database: {e}")
        db.rollback()
    finally:
        db.close()
