"""
This package provides functionality for working block arrays (also called nested arrays)
"""


## Detect if optional packages exist
def make_require(module_name, has_module):
    def require_module(func):
        def dec_func(*args, **kwargs):
            if has_module:
                return func(*args, **kwargs)
            else:
                raise ImportError(
                    f"Function {func} this can't be called without {module_name}"
                )

        return dec_func

    return require_module


try:
    import jax as _
except ImportError as e:
    _HAS_JAX = False
else:
    _HAS_JAX = True
require_jax = make_require('jax', _HAS_JAX)

try:
    from petsc4py import PETSc as _
except ImportError:
    _HAS_PETSC = False
else:
    _HAS_PETSC = True
require_petsc = make_require('petsc4py', _HAS_PETSC)

try:
    import dolfin as _
except ImportError:
    _HAS_FENICS = False
else:
    _HAS_FENICS = True
require_fenics = make_require('fenics', _HAS_FENICS)

try:
    import h5py as _
except ImportError:
    _HAS_H5PY = False
else:
    _HAS_H5PY = True
require_h5py = make_require('h5py', _HAS_H5PY)


from . import linalg
from . import blockmat
from . import blockvec
