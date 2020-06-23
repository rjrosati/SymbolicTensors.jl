TensorHead(s::TensorHead) = s
TensorHead(name::AbstractString,index_types::AbstractArray{TensorIndexType},symmetry::TensorSymmetry)::TensorHead = tensor.TensorHead(name,index_types,symmetry)
TensorHead(name::AbstractString,index_types::AbstractArray{TensorIndexType})::TensorHead = tensor.TensorHead(name,index_types)
TensorHead(name::Symbol,index_types::AbstractArray{TensorIndexType},symmetry::TensorSymmetry)::TensorHead = tensor.TensorHead(name,index_types,symmetry)
TensorHead(name::Symbol,index_types::AbstractArray{TensorIndexType})::TensorHead = tensor.TensorHead(name,index_types)
(t::TensorHead)(ics::TensorIndex...) = t.__pyobject__(ics...)


## Eg. A.norm() where A = [x 1; 1 x], say
function Base.getproperty(A::TensorSymmetry, k::Symbol)
    if k in fieldnames(typeof(A))
        return getfield(A,k)
    else
        M1 = getproperty(PyCall.PyObject(A), k)
        convert(TensorSymmetry,M1)
    end
end


macro heads(IT::AbstractArray{TensorIndexType},xs...)
    hs = []
    for x in xs
        push!(hs,TensorHead(x,IT))
    end
    return hs
end
