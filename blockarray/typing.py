"""
Modules to collect all types used for type hints
"""

from typing import TypeVar, Tuple, List, Union, Mapping

Scalar = Union[int, float]

T = TypeVar("T")
NestedArray = Union['NestedArray', Tuple[T, ...]]
FlatArray = Tuple[T, ...]

Shape = Tuple[int, ...]
Strides = Tuple[int, ...]
BlockShape = Tuple[Tuple[int, ...], ...]

Labels = Tuple[str, ...]
MultiLabels = Tuple[Labels, ...]

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
