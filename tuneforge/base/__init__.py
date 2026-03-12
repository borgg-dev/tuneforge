"""
TuneForge base module — protocol definitions and neuron base classes.
"""

from tuneforge.base.protocol import (
    MusicGenerationSynapse,
    PingSynapse,
    HealthReportSynapse,
)
from tuneforge.base.neuron import BaseNeuron
from tuneforge.base.miner import BaseMinerNeuron
from tuneforge.base.validator import BaseValidatorNeuron
from tuneforge.base.dendrite import DendriteResponseEvent

__all__ = [
    "MusicGenerationSynapse",
    "PingSynapse",
    "HealthReportSynapse",
    "BaseNeuron",
    "BaseMinerNeuron",
    "BaseValidatorNeuron",
    "DendriteResponseEvent",
]
