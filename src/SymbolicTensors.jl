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

struct Tensor <: SymbolicObject end
struct TensorHead <: SymbolicObject
    __pyobject__::PyCall.PyObjct
end
struct TensorIndex <: SymbolicObject
    __pyobject__::PyCall.PyObjct
end
struct TensorIndexType <: SymbolicObject
    __pyobject__::PyCall.PyObjct
end

TensorIndex(s::TensorIndex) == s
TensorIndex(name::AbstractString,tensor_index_type::TensorIndexType,is_up::Bool=true) = sympy.tensor.TensorIndex(name,tensor_index_type,is_up)
TensorIndex(name::Symbol,tensor_index_type::TensorIndexType,is_up::Bool=true) = sympy.tensor.TensorIndex(name,tensor_index_type,is_up)
#-(x::TensorIndex) = x.__neg__()

TensorHead(s::TensorHead) == s
TensorHead(name::AbstractString,index_types::AbstractArray{TensorIndexType},symmetry::TensorSymmetry) = sympy.tensor.TensorHead(name,index_types,symmetry)
TensorHead(name::Symbol,index_types::AbstractArray{TensorIndexType},symmetry::TensorSymmetry) = sympy.tensor.TensorHead(name,index_types,symmetry)

TensorIndexType(name::AbstractString,dummy_name::AbstractString,dim::Int,metric_name::AbstractString) = sympy.tensor.TensorIndexType(name,dummy_name,dim,metric_name=metric_name)
TensorIndexType(name::Symbol,dummy_name::Symbol,dim::Int,metric_name::Symbol) = sympy.tensor.TensorIndexType(string(name),string(dummy_name),dim,metric_name=string(metric_name))

macro heads(TIT::TensorIndexType,x...)

end

macro indices(TIT::TensorIndexType,x...)

end

end
