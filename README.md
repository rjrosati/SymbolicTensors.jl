# SymbolicTensors

[![Build Status](https://travis-ci.com/rjrosati/SymbolicTensors.jl.svg?token=zMDX3GmCZbdBcf9JWMdp&branch=master)](https://travis-ci.com/rjrosati/SymbolicTensors.jl)
[![Codecov](https://codecov.io/gh/rjrosati/SymbolicTensors.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/rjrosati/SymbolicTensors.jl)

Many numerical tensor manipulation packages exist (e.g. `Einsum.jl`), but treating tensors at a purely numeric level throws away a lot of potential optimizations.
Often, it's possible to exploit the symmetries of a problem to dramatically reduce the calculation steps necessary, or perform some tensor contractions symbolically rather than numerically.

`SymbolicTensors.jl` is designed to exploit these simplifications to perform symbolic calculations and generate more efficient input into numeric tensor packages than you would write by hand. It based on `SymPy.jl`, `sympy.tensor.tensor`, `xTensor`, and `ITensors.jl`.

## Rough TODO
* Nicer errors
* testing coverage
* add conversion to ITensors or equivalent
* get TensorSymmetry typed correctly and working
* alias bp_canon to something
* Derivatives!!!
* documentation
