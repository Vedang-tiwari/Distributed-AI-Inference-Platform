import json
import logging
import random
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session

from core.db.session import ModelRegistryEntry, CanaryDeployment

logger = logging.getLogger(__name__)


class ModelRegistryService:
    def list_models(self, db: Session) -> List[Dict]:
        entries = db.query(ModelRegistryEntry).all()
        models_map: Dict[str, Dict] = {}

        for entry in entries:
            name = entry.name
            if name not in models_map:
                models_map[name] = {
                    "name": name,
                    "versions": [],
                    "latest_version": entry.version,
                    "status": entry.status,
                    "canary_weights": None,
                }
            models_map[name]["versions"].append(entry.version)
            models_map[name]["latest_version"] = entry.version  # ordered by creation

        # Attach canary weights if any
        canary_records = db.query(CanaryDeployment).all()
        for c in canary_records:
            if c.model_name in models_map:
                try:
                    models_map[c.model_name]["canary_weights"] = json.loads(c.weights_json)
                except Exception:
                    pass

        return list(models_map.values())

    def get_model_versions(self, db: Session, model_name: str) -> Dict:
        entries = db.query(ModelRegistryEntry).filter(ModelRegistryEntry.name == model_name).all()
        if not entries:
            return {"model": model_name, "versions": []}

        canary = db.query(CanaryDeployment).filter(CanaryDeployment.model_name == model_name).first()
        weights = {}
        if canary:
            try:
                weights = json.loads(canary.weights_json)
            except Exception:
                weights = {}

        return {
            "model": model_name,
            "versions": [e.version for e in entries],
            "details": [
                {
                    "version": e.version,
                    "framework": e.framework,
                    "device": e.device,
                    "status": e.status,
                    "artifact_uri": e.artifact_uri,
                    "created_at": str(e.created_at),
                }
                for e in entries
            ],
            "canary_weights": weights,
        }

    def resolve_version(self, db: Session, model_name: str, requested_version: Optional[str] = None) -> Tuple[str, str]:
        """
        Resolves the exact version and required device ('cpu' or 'gpu') to use.
        If requested_version is provided, validates and returns it.
        Otherwise, uses canary routing weights or picks the latest version.
        """
        # 1. If explicit version specified
        if requested_version:
            entry = db.query(ModelRegistryEntry).filter(
                ModelRegistryEntry.name == model_name,
                ModelRegistryEntry.version == requested_version,
            ).first()
            if entry:
                return entry.version, entry.device
            # Default to requested version on cpu if not found
            return requested_version, "cpu"

        # 2. Check Canary routing weights
        canary = db.query(CanaryDeployment).filter(CanaryDeployment.model_name == model_name).first()
        if canary and canary.weights_json:
            try:
                weights: Dict[str, float] = json.loads(canary.weights_json)
                if weights:
                    versions = list(weights.keys())
                    probs = list(weights.values())
                    selected_version = random.choices(versions, weights=probs, k=1)[0]
                    entry = db.query(ModelRegistryEntry).filter(
                        ModelRegistryEntry.name == model_name,
                        ModelRegistryEntry.version == selected_version,
                    ).first()
                    device = entry.device if entry else "cpu"
                    return selected_version, device
            except Exception as e:
                logger.warning("Canary resolution fallback due to: %s", e)

        # 3. Fallback to latest READY version
        latest_entry = db.query(ModelRegistryEntry).filter(
            ModelRegistryEntry.name == model_name,
            ModelRegistryEntry.status == "READY",
        ).order_by(ModelRegistryEntry.id.desc()).first()

        if latest_entry:
            return latest_entry.version, latest_entry.device

        return "v1", "cpu"

    def register_model(
        self,
        db: Session,
        name: str,
        version: str,
        framework: str = "pytorch",
        device: str = "cpu",
        artifact_uri: str = "",
        description: str = "",
    ) -> ModelRegistryEntry:
        existing = db.query(ModelRegistryEntry).filter(
            ModelRegistryEntry.name == name,
            ModelRegistryEntry.version == version,
        ).first()

        if existing:
            existing.framework = framework
            existing.device = device
            existing.artifact_uri = artifact_uri
            existing.description = description
            existing.status = "READY"
            db.commit()
            db.refresh(existing)
            return existing

        new_entry = ModelRegistryEntry(
            name=name,
            version=version,
            framework=framework,
            device=device,
            artifact_uri=artifact_uri,
            description=description,
            status="READY",
        )
        db.add(new_entry)
        db.commit()
        db.refresh(new_entry)
        return new_entry

    def set_canary_weights(self, db: Session, model_name: str, weights: Dict[str, float]) -> Dict[str, float]:
        canary = db.query(CanaryDeployment).filter(CanaryDeployment.model_name == model_name).first()
        if not canary:
            canary = CanaryDeployment(model_name=model_name, weights_json=json.dumps(weights))
            db.add(canary)
        else:
            canary.weights_json = json.dumps(weights)
        db.commit()
        return weights


model_registry_service = ModelRegistryService()
