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

function Base.:+(A::Number, B::TensScalar)
    return TensScalar(B.var,B.expr+A,B.__pyobject__)
end
function Base.:-(A::Number, B::TensScalar)
    return TensScalar(B.var,B.expr-A,B.__pyobject__)
end
function Base.:/(A::Number, B::TensScalar)
    return TensScalar(B.var,A/B.expr,B.__pyobject__)
end
function Base.:/(A::TensScalar,B::Number)
    return TensScalar(A.var,A.expr/B,A.__pyobject__)
end
function Base.:*(A::TensScalar,B::Number)
    return TensScalar(A.var,A.expr*B,A.__pyobject__)
end

function Base.:*(A::TensScalar,B::T) where {T <: Tensor}
    return Base.:*(B,A)
end
function Base.:*(B::T,A::TensScalar) where {T <:Tensor}
    return TensMul(A.expr*B)
end
function Base.:^(A::TensScalar,B::Number)
    return TensScalar(A.var,A.expr^B,A.__pyobject__)
end
function Base.:^(A::TensScalar,B::Integer)
    return TensScalar(A.var,A.expr^B,A.__pyobject__)
end
function Base.:^(A::Number,B::TensScalar)
    return TensScalar(B.var,A^B.expr,B.__pyobject__)
end
function Base.log(B::TensScalar)
    return TensScalar(B.var,log(B.expr),B.__pyobject__)
end

function Base.show(io::IO, ::MIME"text/plain", t::TensMul)
    Base.show(io,::MIME"text/plain",convert(t.__pyobject__,SymbolicObject))
    print(io, sympy.pretty(t.expr))
    println(io,"")
    println(io,"")
    print(io, sympy.pretty(sympy.Eq(t.var,t.__pyobject__)))
    return nothing
function Base.show(io::IO,  t::TensScalar)
    show(io,t.expr)
    print(io,",")
    print(io, jprint(sympy.Eq(t.var,t.__pyobject__)))
    return nothing
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
    global scalars
    pyexp = A.__pyobject__*B.__pyobject__
    inds = pycall(getproperty(pyexp,:get_free_indices),Array{TensorIndex})
    if length(inds) == 0
        x = get_scalar()
        return TensScalar(x,x,pyexp)
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
