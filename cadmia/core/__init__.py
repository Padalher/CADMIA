"""cadmia.core — unified data model (the shared spine of the CADMIA suite).

Structural/electronic types (Structure, ElectronicState, VolumetricGrid) plus the orthogonal
model classes (Provenance, Descriptor/DescriptorSet, AdsorptionSystem, Calculation) and adapters
from the existing agent results. Parsers live in cadmia.io.
"""
from cadmia.core.atom import Atom
from cadmia.core.bond import Bond
from cadmia.core.cell import Cell
from cadmia.core.structure import Structure
from cadmia.core.electronic_state import ElectronicState
from cadmia.core.grid import VolumetricGrid
from cadmia.core.provenance import Provenance
from cadmia.core.descriptor import Descriptor, DescriptorSet
from cadmia.core.adsorption import AdsorptionSystem
from cadmia.core.calculation import Calculation

__all__ = [
    "Atom", "Bond", "Cell", "Structure", "ElectronicState", "VolumetricGrid",
    "Provenance", "Descriptor", "DescriptorSet", "AdsorptionSystem", "Calculation",
]
