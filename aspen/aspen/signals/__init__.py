"""
SignalsDummy package that sets out a structure for defining,
normalising & combining signal objects
"""

from aspen.signals.core import ISignal, ISignals, INormalise
from aspen.signals.leaf import ILeaf, Leaf, LeafSeries
from aspen.signals.generic import SignalHeap
