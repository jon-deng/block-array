"""
Modules to collect all types used for type hints
"""

from typing import TypeVar, Tuple, Union, Mapping

T = TypeVar("T")

NestedArray = Union['NestedArray', Tuple[T, ...]]
FlatArray = Tuple[T, ...]

Shape = Tuple[int, ...]
Strides = Tuple[int, ...]

Labels = Tuple[str, ...]
MultiLabels = Tuple[Labels, ...]

## Indexing types
IntIndex = int
IntIndices = Tuple[int, ...]
EllipsisType = type(...)

# General and Standard index
GenIndex = Union[slice, IntIndex, str, IntIndices, EllipsisType]
StdIndex = Union[IntIndex, IntIndices]

MultiStdIndex = Tuple[StdIndex, ...]
MultiGenIndex = Tuple[GenIndex, ...]

LabelToIntIndex = Mapping[str, IntIndex]
MultiLabelToIntIndex = Tuple[LabelToIntIndex, ...]