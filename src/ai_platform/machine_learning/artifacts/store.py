"""Save and load trusted AI model artifacts."""

from pathlib import Path

import joblib


class ModelArtifactStore:
    """Store versioned model files outside the source code."""

    def __init__(
        self,
        root: str | Path = Path("artifacts/models"),
    ) -> None:
        self.root = Path(root)

    def _resolve_path(self, artifact_uri: str | Path) -> Path:
        """Resolve an artifact path and prevent access outside the store."""
        root = self.root.resolve()
        supplied_path = Path(artifact_uri)
        candidate = (
            supplied_path.resolve()
            if supplied_path.is_absolute()
            else (root / supplied_path).resolve()
        )

        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "Artifact path must be inside the model store."
            ) from exc

        if candidate.suffix != ".joblib":
            raise ValueError("Model artifacts must use the .joblib extension.")

        return candidate

    def save(
        self,
        model: object,
        model_id: int,
        version: str,
    ) -> Path:
        """Save a model using its registry ID and version."""
        if isinstance(model_id, bool) or model_id <= 0:
            raise ValueError("model_id must be a positive integer.")

        if not version.strip() or version in {".", ".."}:
            raise ValueError("version must be a valid non-empty value.")

        if "/" in version or "\\" in version:
            raise ValueError("version must not contain path separators.")

        destination = self._resolve_path(
            Path(str(model_id)) / f"{version}.joblib"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, destination)

        return destination

    def load(self, artifact_uri: str | Path) -> object:
        """Load a trusted model artifact from the store."""
        path = self._resolve_path(artifact_uri)

        if not path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {path}")

        # joblib loading can execute code; only load trusted artifacts.
        return joblib.load(path)
