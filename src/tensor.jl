# should probably port this from sympy.tensor.tensor, but works with SymPy.jl
using SymPy
using PyCall

import Base: show
import Base: convert, promote_rule
import Base: getproperty
import Base: hash, ==
import Base: length, size
import Base.iterate
import Base: +, -, *, /, //, \, ^

"Docstring..."
struct TensorIndexType
    name::String
    dummy_fmt::String
    dim::Int
    metric_symmetry::Bool
    metric_name::String
end
