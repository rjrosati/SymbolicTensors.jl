module SymbolicTensors

using SymPy
using PyCall

import SymPy: SymbolicObject,Sym,diff

import Base: show
import Base: convert, promote_rule
import Base: getproperty
import Base: hash, ==
import Base: length, size
import Base.iterate
import Base: +, -, *, /, //, \, ^

export TensorIndexType, tensor_indices, TensorIndex, TensorHead
export IndexedTensor, TensAdd, TensMul, TensScalar
export TensorSymmetry
#export @heads,@indices
export @indices
export tensor
export diff

abstract type Tensor <: SymbolicObject end
struct TensorIndex <: SymbolicObject
    __pyobject__::PyCall.PyObject
end
struct TensorIndexType <: SymbolicObject
    __pyobject__::PyCall.PyObject
end
struct TensorSymmetry <: SymbolicObject
    __pyobject__::PyCall.PyObject
end
struct TensorHead <: SymbolicObject
    __pyobject__::PyCall.PyObject
end

struct TensScalar <: Tensor
    __pyobject__::PyCall.PyObject;
    expr::Sym;
end

struct TensMul <: Tensor
    __pyobject__::PyCall.PyObject
end
struct TensAdd <: Tensor
    __pyobject__::PyCall.PyObject
end
struct IndexedTensor <: Tensor
    __pyobject__::PyCall.PyObject
end

const tensor = PyCall.PyNULL()
const toperations = PyCall.PyNULL()

function __init__()
    copy!(tensor,PyCall.pyimport_conda("sympy.tensor.tensor","sympy"))
    #pytype_mapping(tensor.TensorSymmetry,TensorSymmetry)
#    copy!(toperations,PyCall.pyimport_conda("sympy.tensor.toperations","sympy"))
end

include("tensor.jl")
include("indices.jl")
include("derivatives.jl")

end
