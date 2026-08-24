"""Visual token compression.

``qsampler`` compresses the connector output in LM embedding space against a
target-invariance objective (prior work, not on the drafting path).
``row_compressor`` + ``splice`` compress the target aux hidden states the EAGLE-3
drafter actually consumes, trained by the draft loss alone.
"""

from .qsampler import QSampler, QSamplerConfig
from .row_compressor import VisRowCompressor, VisRowCompressorConfig
from .splice import compress_image_rows

__all__ = [
    "QSampler",
    "QSamplerConfig",
    "VisRowCompressor",
    "VisRowCompressorConfig",
    "compress_image_rows",
]
