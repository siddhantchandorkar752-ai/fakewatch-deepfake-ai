from .fakewatch import FakeWatch
from .spatial import SpatialExtractor
from .temporal import TemporalTransformer
from .fusion import FusionModule
from .classifier import DeepfakeClassifier

__all__ = [
    "FakeWatch",
    "SpatialExtractor",
    "TemporalTransformer",
    "FusionModule",
    "DeepfakeClassifier",
]
