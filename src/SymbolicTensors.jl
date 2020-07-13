module SymbolicTensors

using SymPy
using PyCall

import SymPy: SymbolicObject,Sym,diff
import SymPy: jprint,factor
import SymPy: _convert

import Base: show
import Base: convert, promote_rule
import Base: getproperty
import Base: hash, ==
import Base: length, size
import Base.iterate
import Base: +, -, *, /, //, \, ^, log

export TensorIndexType, tensor_indices, TensorIndex, TensorHead
export IndexedTensor, TensAdd, TensMul, TensScalar
export TensorSymmetry
#export @heads,@indices
export @indices, @heads
export tensor
export diff
export canon_bp, contract_metric, factor
export get_tsymmetry
export scalarIsEqual
export replace_with_arrays
export cse


const tensor = PyCall.PyNULL()
const toperations = PyCall.PyNULL()
const get_tsymmetry = convert(TensorSymmetry,PyCall.PyNULL())

function __init__()
    copy!(tensor,PyCall.pyimport_conda("sympy.tensor.tensor","sympy"))
    copy!(get_tsymmetry.__pyobject__,tensor.TensorSymmetry)
    #pytype_mapping(tensor.TensorSymmetry,TensorSymmetry)
#    copy!(toperations,PyCall.pyimport_conda("sympy.tensor.toperations","sympy"))
end

include("types.jl")
include("tensor.jl")
include("indices.jl")
include("derivatives.jl")
include("quoting.jl")

end
