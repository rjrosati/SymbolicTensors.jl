module SymbolicTensors

using SymPy
using PyCall

import SymPy: SymbolicObject,Sym

import Base: show
import Base: convert, promote_rule
import Base: getproperty
import Base: hash, ==
import Base: length, size
import Base.iterate
import Base: +, -, *, /, //, \, ^

export TensorIndexType, tensor_indices, TensorIndex, TensorHead
export TensorSymmetry
#export @heads,@indices
export @indices
export tensor

struct Tensor <: SymbolicObject end
struct TensorHead <: SymbolicObject
    __pyobject__::PyCall.PyObject
end
struct TensorIndex <: SymbolicObject
    __pyobject__::PyCall.PyObject
end
struct TensorIndexType <: SymbolicObject
    __pyobject__::PyCall.PyObject
end
struct TensorSymmetry <: SymbolicObject
    __pyobject__::PyCall.PyObject
end

const tensor = PyCall.PyNULL()
const toperations = PyCall.PyNULL()

function __init__()
    copy!(tensor,PyCall.pyimport_conda("sympy.tensor.tensor","sympy"))
#    copy!(toperations,PyCall.pyimport_conda("sympy.tensor.toperations","sympy"))
end

TensorIndex(s::TensorIndex) = s
TensorIndex(name::AbstractString,tensor_index_type::TensorIndexType,is_up::Bool=true)::TensorIndex = tensor.TensorIndex(name,tensor_index_type,is_up)
TensorIndex(name::Symbol,tensor_index_type::TensorIndexType,is_up::Bool=true)::TensorIndex = tensor.TensorIndex(name,tensor_index_type,is_up)

-( x::TensorIndex ) = TensorIndex(x.__neg__())

TensorHead(s::TensorHead) = s
TensorHead(name::AbstractString,index_types::AbstractArray{TensorIndexType},symmetry::TensorSymmetry)::TensorHead = tensor.TensorHead(name,index_types,symmetry)
TensorHead(name::AbstractString,index_types::AbstractArray{TensorIndexType})::TensorHead = tensor.TensorHead(name,index_types)
TensorHead(name::Symbol,index_types::AbstractArray{TensorIndexType},symmetry::TensorSymmetry)::TensorHead = tensor.TensorHead(name,index_types,symmetry)
TensorHead(name::Symbol,index_types::AbstractArray{TensorIndexType})::TensorHead = tensor.TensorHead(name,index_types)
(t::TensorHead)(ics::TensorIndex...) = t.__pyobject__(ics...)


TensorIndexType(name::AbstractString,dummy_fmt::AbstractString,dim::Int,metric_name::AbstractString)::TensorIndexType = tensor.TensorIndexType(name,dummy_fmt=dummy_fmt,dim,metric_name)
TensorIndexType(name::AbstractString,dummy_fmt::AbstractString,dim::Int)::TensorIndexType = tensor.TensorIndexType(name,dummy_fmt=dummy_fmt,dim=dim)
TensorIndexType(name::AbstractString,dummy_fmt::AbstractString)::TensorIndexType = tensor.TensorIndexType(name,dummy_fmt=dummy_fmt)
TensorIndexType(name::Symbol,dummy_name::Symbol,dim::Int,metric_name::Symbol)::TensorIndexType = tensor.TensorIndexType(string(name),string(dummy_name),dim,string(metric_name))


macro heads(IT::AbstractArray{TensorIndexType},xs...)
    hs = []
    for x in xs
        push!(hs,TensorHead(x,IT))
    end
    return hs
end
macro heads(IT::AbstractArray{TensorIndexType},xs...)
    hs = []
    for x in xs
        push!(hs,TensorHead(x,IT))
    end
    return hs
end

macro indices(TIT::TensorIndexType,x...)

end

end
