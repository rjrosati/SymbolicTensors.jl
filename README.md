# SymbolicTensors

[![Build Status](https://travis-ci.com/rjrosati/SymbolicTensors.jl.svg?token=zMDX3GmCZbdBcf9JWMdp&branch=master)](https://travis-ci.com/rjrosati/SymbolicTensors.jl)
[![codecov](https://codecov.io/gh/rjrosati/SymbolicTensors.jl/branch/master/graph/badge.svg?token=JKgibtSJzc)](https://codecov.io/gh/rjrosati/SymbolicTensors.jl)



Many numerical tensor manipulation packages exist (e.g. `Einsum.jl`), but treating tensors at a purely numeric level throws away a lot of potential optimizations.
Often, it's possible to exploit the symmetries of a problem to dramatically reduce the calculation steps necessary, or perform some tensor contractions symbolically rather than numerically.

`SymbolicTensors.jl` is designed to exploit these simplifications to perform symbolic calculations and generate more efficient input into numeric tensor packages than you would write by hand. It based on `SymPy.jl`, `sympy.tensor.tensor`, `xTensor`, and `ITensors.jl`.

See the talk about this package given at JuliaCon 2020: https://www.youtube.com/watch?v=_b4JIv044GY

## Example calculations
```julia
using SymbolicTensors
using SymPy

spacetime = TensorIndexType("spacetime","f")
@indices spacetime μ ν σ ρ η
x = TensorHead("x",[spacetime])
δ = spacetime.delta
# one way to write the metric on a sphere
g = 4*δ(-μ,-ν)/(1+x(μ)*x(ν)*δ(-μ,-ν))^2

# compute the christoffel symbols
Γ = (diff(g(-μ,-ν),x(σ)) - diff(g(-ν,-σ),x(μ)) + diff(g(-σ,-μ),x(ν)))/2
Γ = factor(contract_metric(canon_bp(Γ),spacetime.metric))
```
At this point `Γ` is a symbolic tensor expression, written in Einstein notation.
It could represent 2 dimensional or 10,000 dimensional tensors in the same form -- this is one of the great advantages of working at the symbolic tensor level.

We could manipulate `Γ` further, or convert it into an array.
```julia
# convert Γ to Array{Sym}
xarr = symbols("x y",real=true)
garr = replace_with_arrays(g,Dict(x(ρ) => xarr, spacetime.delta(-ρ,-η) => [1 0; 0 1]))
Γarr = replace_with_arrays(Γ,Dict(x(ρ) => xarr, spacetime.delta(-ρ,-η) => [1 0; 0 1], spacetime => garr))
```
Now that we have an array in `Γarr`, `Quote` allows us to convert it into a native Julia function.
```julia
# Then Quote it and eval to a Julia function
quot = Quote("Christoffel",Γarr,[ x for x in xarr])
Christ = eval(quot)
Christ(0,1)
```

## Rough TODO
* Nicer errors
* testing coverage
* add conversion to ITensors or equivalent
* documentation
* symbolic derivatives?

## known bugs
* check issues
