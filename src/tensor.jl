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


## Eg. A.norm() where A = [x 1; 1 x], say
function Base.getproperty(A::TensorSymmetry, k::Symbol)
    if k in fieldnames(typeof(A))
        return getfield(A,k)
    else
        M1 = getproperty(PyCall.PyObject(A), k)
        convert(TensorSymmetry,M1)
    end
end

TensAdd(s::TensAdd) = s
TensMul(s::TensMul) = s
function terms(s::T) where T <: Tensor
    sp = getproperty(PyCall.PyObject(s),:split)
    return IndexedTensor[ x for x in pycall(sp,Array{IndexedTensor})]
end
function Base.:+(A::T ,B::T) where {T <: Tensor}
    pyexp = convert(TensAdd,A.__pyobject__+B.__pyobject__)
    return pyexp
end
function Base.:*(A::T ,B::T) where {T <: Tensor}
    return convert(TensMul,A.__pyobject__*B.__pyobject__)
end

macro heads(IT::AbstractArray{TensorIndexType},xs...)
    hs = []
    for x in xs
        push!(hs,TensorHead(x,IT))
    end
    return hs
end
