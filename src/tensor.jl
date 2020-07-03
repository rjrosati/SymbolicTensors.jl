TensorHead(s::TensorHead) = s
TensorHead(name::AbstractString,index_types::AbstractArray{TensorIndexType},symmetry::TensorSymmetry)::TensorHead = tensor.TensorHead(name,index_types,symmetry)
TensorHead(name::AbstractString,index_types::AbstractArray{TensorIndexType})::TensorHead = tensor.TensorHead(name,index_types)
TensorHead(name::Symbol,index_types::AbstractArray{TensorIndexType},symmetry::TensorSymmetry)::TensorHead = tensor.TensorHead(name,index_types,symmetry)
TensorHead(name::Symbol,index_types::AbstractArray{TensorIndexType})::TensorHead = tensor.TensorHead(name,index_types)
(t::TensorHead)(ics::TensorIndex...) = convert(IndexedTensor,t.__pyobject__(ics...))


function Base.getproperty(A::TensorHead, k::Symbol)
    if k == :index_types
        m = getproperty(PyCall.PyObject(A),k)
        p = PyCall.PyObject(m)
        return TensorIndexType[ x for x in p]
    elseif k in fieldnames(typeof(A))
        return getfield(A,k)
    # else error?
    end
end

function Base.getproperty(A::TensMul, k::Symbol)
    if k == :coeff
        m = getproperty(PyCall.PyObject(A),k)
        return m
    elseif k == :nocoeff
        m = getproperty(PyCall.PyObject(A),k)
        return convert(IndexedTensor,m)
    elseif k in fieldnames(typeof(A))
        return getfield(A,k)
    # else error?
    end
end

## Eg. A.norm() where A = [x 1; 1 x], say
function Base.getproperty(A::TensorSymmetry, k::Symbol)
    if k in fieldnames(typeof(A))
        return getfield(A,k)
    else
        M1 = getproperty(PyCall.PyObject(A), k)
        convert(TensorSymmetry,M1)
    end
end
scalar_name = "ts"
scalar_index = 1
function get_scalar()
    global scalar_name,scalar_index
    symb = symbols(scalar_name * string(scalar_index),positive=true)
    scalar_index += 1
    return symb
end


TensAdd(s::TensAdd) = s
TensMul(s::TensMul) = s
function terms(s::T) where T <: Tensor
    sp = getproperty(PyCall.PyObject(s),:split)
    ret = []
    for x in pycall(sp,PyCall.PyObject)
        if x.__class__.__name__ == "TensMul"
            push!(ret,convert(TensMul,x))
        else
            push!(ret,convert(IndexedTensor,x))
        end
    end
    return ret
end
function Base.:+(A::T ,B::U) where {T <: Tensor, U <: Tensor}
    pyexp =  A.__pyobject__+B.__pyobject__
    if pyexp.__class__.__name__ == "TensMul"
        return convert(TensMul,pyexp)
    else
        return convert(TensAdd,pyexp)
    end
end
function Base.:*(A::T ,B::U) where {T <: Tensor, U <: Tensor}
    pyexp = A.__pyobject__*B.__pyobject__
    inds = pycall(getproperty(pyexp,:get_free_indices),Array{TensorIndex})
    if length(inds) == 0
        return TensScalar(pyexp,get_scalar())
    else
        return convert(TensMul,pyexp)
    end
end
function Base.:*(A::T ,B::U) where {T <: Real, U <: Tensor}
    pyexp = A*B.__pyobject__
    return convert(TensMul,pyexp)
end

macro heads(IT::AbstractArray{TensorIndexType},xs...)
    hs = []
    for x in xs
        push!(hs,TensorHead(x,IT))
    end
    return hs
end
