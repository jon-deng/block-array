"""
Modules to collect all types used for type hints
"""
from typing import TypeVar, Tuple, List, Union, Mapping, Any

from . import _HAS_PETSC, _HAS_FENICS, _HAS_JAX
if _HAS_PETSC:
    from petsc4py import PETSc
if _HAS_FENICS:
    import dolfin as dfn
if _HAS_JAX:
    from jax import numpy as jnp

# Special vector/matrix types
_Null = type(None)
if _HAS_PETSC:
    PETScMat = PETSc.Mat
    PETScVec = PETSc.Vec
else:
    PETScMat = _Null
    PETScVec = _Null

if _HAS_FENICS:
    DfnMat = dfn.PETScMatrix
    DfnVec = dfn.PETScVector
else:
    DfnMat = _Null
    DfnVec = _Null

if _HAS_JAX:
    JaxArray = jnp.ndarray
else:
    JaxArray = _Null

Scalar = Union[int, float]

T = TypeVar("T")
NestedArray = Union['NestedArray', Tuple[T, ...]]
FlatArray = Tuple[T, ...]

Shape = Tuple[int, ...]
Strides = Tuple[int, ...]
BlockShape = Tuple[Tuple[int, ...], ...]

Labels = Tuple[str, ...]
MultiLabels = Tuple[Labels, ...]

## BlockShape type
AxisSize = Tuple[Union[int, 'AxisSize'], ...]

## Indexing types
# These types represent an index to single element
StdIndex = int
GenIndex = Union[int, str]

# These types represents indexes to a collection of elements
GenIndices = List[GenIndex]
# a slice also selects a collection of elements

StdIndices = List[StdIndex]

# Special type for expanding missing indices
EllipsisType = type(...)

# Multidimensions general/standard indices
MultiGenIndex = Tuple[Union[GenIndex, GenIndices, slice, EllipsisType], ...]
MultiStdIndex = Tuple[Union[StdIndex, StdIndices], ...]

# These types represent the mapping from labels to indices
LabelToStdIndex = Mapping[str, StdIndex]
MultiLabelToStdIndex = Tuple[LabelToStdIndex, ...]
