
TensorIndex(s::TensorIndex) = s
TensorIndex(name::AbstractString,tensor_index_type::TensorIndexType,is_up::Bool=true)::TensorIndex = tensor.TensorIndex(name,tensor_index_type,is_up)
TensorIndex(name::Symbol,tensor_index_type::TensorIndexType,is_up::Bool=true)::TensorIndex = tensor.TensorIndex(name,tensor_index_type,is_up)
-( x::TensorIndex ) = TensorIndex(x.__neg__())

TensorIndexType(name::AbstractString,dummy_name::AbstractString,dim::Int,metric_name::AbstractString)::TensorIndexType = tensor.TensorIndexType(name,dummy_name=dummy_name,dim=dim,metric_name=metric_name)
TensorIndexType(name::AbstractString,dummy_name::AbstractString,dim::Int)::TensorIndexType = tensor.TensorIndexType(name,dummy_name=dummy_name,dim=dim)
TensorIndexType(name::AbstractString,dummy_name::AbstractString)::TensorIndexType = tensor.TensorIndexType(name,dummy_name=dummy_name)
TensorIndexType(name::Symbol,dummy_name::Symbol,dim::Int,metric_name::Symbol)::TensorIndexType = tensor.TensorIndexType(string(name),string(dummy_name),dim,string(metric_name))


function Base.getproperty(A::IndexedTensor, k::Symbol)
    if k == :head
        m = getproperty(PyCall.PyObject(A),k)
        return convert(TensorHead,m)
    #elseif k == :indices
    #    m = getproperty(PyCall.PyObject(A),:indices)
    #    println(m)
    #    return convert(Array{TensorIndex},m)
    elseif k == :free
        m = getproperty(PyCall.PyObject(A),k)
        return convert(Array{Tuple{TensorIndex,Int}},m)
    elseif k == :free_indices
        m = pycall(getproperty(PyCall.PyObject(A),:get_free_indices),Array{TensorIndex})
        return m
    elseif k in fieldnames(typeof(A))
        return getfield(A,k)
    else
        M1 = getproperty(PyCall.PyObject(A), k)
        M1
    end
end

function Base.getproperty(A::TensorIndex, k::Symbol)
    if k == :is_up
        m = getproperty(PyCall.PyObject(A),k)
        return convert(Bool,m)
    elseif k in fieldnames(typeof(A))
        return getfield(A,k)
    else
        M1 = getproperty(PyCall.PyObject(A), k)
        M1
    end
end
function Base.getproperty(A::T, k::Symbol) where {T <: Tensor}
    if k == :free
        m = getproperty(PyCall.PyObject(A),k)
        return convert(Array{Tuple{TensorIndex,Int}},m)
    elseif k in fieldnames(typeof(A))
        return getfield(A,k)
    else
        M1 = getproperty(PyCall.PyObject(A), k)
        M1
    # else error?
    end
end

function Base.getproperty(A::TensorIndexType, k::Symbol)
    if k == :delta || k == :metric
        m = getproperty(PyCall.PyObject(A),k)
        return convert(TensorHead,m)
    elseif k == :set_metric
        m = getproperty(PyCall.PyObject(A),k)
        return m
    elseif k in fieldnames(typeof(A))
        return getfield(A,k)
    else
        M1 = getproperty(PyCall.PyObject(A), k)
        M1
    end
end

macro indices(xs...)
    ret = Expr(:block)
    TIT = xs[1]
    for x in xs[2:end]
        push!(ret.args,:($(esc(x)) = TensorIndex($(string(x)),$(esc(TIT)))))
    end
    return ret
end
