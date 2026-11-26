from abc import ABC, abstractmethod

from app.schemas import VisualMaterial


class VisionProvider(ABC):
    @abstractmethod
    def analyze_material(
        self,
        file_path: str,
        material_type: str = "slide",
        timestamp: float | None = None,
    ) -> VisualMaterial:
        pass
