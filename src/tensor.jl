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
        return sympy_type_convert(m)
    elseif k == :nocoeff
        m = getproperty(PyCall.PyObject(A),k)
        return sympy_type_convert(m)
    elseif k in fieldnames(typeof(A))
        return getfield(A,k)
    # else error?
    end
end

scalar_exprs = Dict{SymPy.Sym,TensMul}()
scalar_names = Dict{TensMul,SymPy.Sym}()
scalar_name = "ts"
scalar_index = 1
function new_scalar(expr::TensMul)
    global scalar_exprs,scalar_names,scalar_index
    if !(expr in keys(scalar_names))
        name = scalar_name * string(scalar_index)
        symb = symbols(name,positive=true)
        scalar_exprs[symb] = expr
        scalar_names[expr] = symb
        scalar_index += 1
        return symb
    else
        return scalar_names[expr]
    end
end
function get_scalars(texpr::T)  where {T <: SymbolicObject}
    vars = [var for var in free_symbols(texpr) if startswith(string(var),scalar_name)]
    return vars
end


#function Base.:+(A::Number, B::TensScalar)
#    return TensScalar(B.var,B.expr+A,B.__pyobject__)
#end
#function Base.:-(A::Number, B::TensScalar)
#    return TensScalar(B.var,B.expr-A,B.__pyobject__)
#end
#function Base.:/(A::Number, B::TensScalar)
#    return TensScalar(B.var,A/B.expr,B.__pyobject__)
#end
#function Base.:/(A::TensScalar,B::Number)
#    return TensScalar(A.var,A.expr/B,A.__pyobject__)
#end
#function Base.:*(A::TensScalar,B::Number)
#    return TensScalar(A.var,A.expr*B,A.__pyobject__)
#end
#
#function Base.:*(A::TensScalar,B::T) where {T <: Tensor}
#    return Base.:*(B,A)
#end
#function Base.:*(B::T,A::TensScalar) where {T <:Tensor}
#    return TensMul(A.expr*B)
#end
#function Base.:^(A::TensScalar,B::Number)
#    return TensScalar(A.var,A.expr^B,A.__pyobject__)
#end
#function Base.:^(A::TensScalar,B::Integer)
#    return TensScalar(A.var,A.expr^B,A.__pyobject__)
#end
#function Base.:^(A::Number,B::TensScalar)
#    return TensScalar(B.var,A^B.expr,B.__pyobject__)
#end
#function Base.log(B::TensScalar)
#    return TensScalar(B.var,log(B.expr),B.__pyobject__)
#end
#
function Base.show(io::IO, ::MIME"text/plain", t::T) where {T <: Tensor}
    print(io, sympy.pretty(t.__pyobject__))
    sc = get_scalars(t)
    if length(sc) > 0
        println(io,"")
        for s in sc
            print(io, sympy.pretty(sympy.Eq(s,scalar_exprs[s])))
            println(io,"")
        end
    end
    return nothing
end
function Base.show(io::IO, ::MIME"text/latex", t::T) where {T <: Tensor}
    print(io, sympy.latex(t.__pyobject__, mode="equation*"))
    sc = get_scalars(t)
    if length(sc) > 0
        println(io,"")
        for s in sc
            print(io, sympy.latex(sympy.Eq(s,scalar_exprs[s]), mode="equation*"))
            println(io,"")
        end
    end
    return nothing
end
function Base.show(io::IO, ::MIME"text/plain", s::TensorSymmetry)
    if s.__pyobject__.__class__.__name__ == "TensorSymmetry"
        print(io, pycall(s.__pyobject__.__repr__,String))
    else
        print(io, s.__pyobject__)
    end
end
#=
function Base.show(io::IO, ::MIME"text/plain", s::SymbolicObject)
    print(io, sympy.pretty(s))
    sc = get_scalars(s)
    if length(sc) > 0
        println(io,"")
        for s in sc
            print(io, sympy.pretty(sympy.Eq(s,scalar_exprs[s])))
            println(io,"")
        end
    end
end
function Base.show(io::IO, ::MIME"text/latex", s::SymbolicObject)
    print(io, sympy.latex(s, mode="equation*"))
    sc = get_scalars(s)
    if length(sc) > 0
        println(io,"")
        for s in sc
            print(io, sympy.latex(sympy.Eq(s,scalar_exprs[s]), mode="equation*"))
            println(io,"")
        end
    end
end
=#
#function Base.show(io::IO,  t::TensScalar)
#    show(io,t.expr)
#    print(io,",")
#    print(io, jprint(sympy.Eq(t.var,t.__pyobject__)))
#    return nothing
#end

TensAdd(s::TensAdd) = s
TensMul(s::TensMul) = s
(t::TensMul)(ics::TensorIndex...) = convert(TensMul,t.__pyobject__(ics...))
(t::TensAdd)(ics::TensorIndex...) = convert(TensAdd,t.__pyobject__(ics...))
(t::IndexedTensor)(ics::TensorIndex...) = convert(IndexedTensor,t.__pyobject__(ics...))
function terms(s::T) where T <: Union{TensAdd,TensMul}
    sp = getproperty(PyCall.PyObject(s),:args)
    ret = []
    for x in sp
        push!(ret,sympy_type_convert(x))
    end
    return ret
end


function Base.:+(A::T ,B::U) where {T <: Tensor, U <: SymbolicObject}
    pyexp =  A.__pyobject__+B.__pyobject__
    return sympy_type_convert(pyexp)
end
function Base.:-(A::T ,B::U) where {T <: Tensor, U <: SymbolicObject}
    pyexp =  A.__pyobject__-B.__pyobject__
    return sympy_type_convert(pyexp)
end
## Eg. A.norm() where A = [x 1; 1 x], say
function Base.getproperty(A::TensorSymmetry, k::Symbol)
    if k == :fully_symmetric || k == :direct_product || k == :no_symmetry
        return x->pycall(getproperty(A.__pyobject__,k), TensorSymmetry, x)
    elseif k == :riemann
        return pycall(getproperty(A.__pyobject__,k), TensorSymmetry)
    elseif k in fieldnames(typeof(A))
        return getfield(A,k)
    end
end

function Base.:*(A::T ,B::U) where {T <: Tensor, U <: Tensor}
    pyexp = A.__pyobject__*B.__pyobject__
    inds = pycall(getproperty(pyexp,:get_free_indices),Array{TensorIndex})
    if length(inds) == 0
        x = new_scalar(convert(TensMul,pyexp))
        return x
    else
        return convert(TensMul,pyexp)
    end
end
function Base.:*(A::T ,B::U) where {T <: Number, U <: Tensor}
    pyexp = A*B.__pyobject__
    return convert(TensMul,pyexp)
end
function Base.:*(A::T ,B::U) where {T <: Tensor, U <: Number}
    pyexp = B*A.__pyobject__
    return convert(TensMul,pyexp)
end
function Base.:*(A::T ,B::U) where {T <: SymbolicObject, U <: Tensor}
    pyexp = A*B.__pyobject__
    return convert(TensMul,pyexp)
end
function Base.:*(A::T ,B::U) where {T <: Tensor, U <: SymbolicObject}
    pyexp = B*A.__pyobject__
    return convert(TensMul,pyexp)
end
function Base.:/(B::U,A::T) where {T <: Number, U <: Tensor}
    pyexp = B.__pyobject__/A
    return sympy_type_convert(pyexp)
end
function Base.:/(B::U,A::T) where {T <: SymbolicObject, U <: Tensor}
    pyexp = B.__pyobject__/A
    return sympy_type_convert(pyexp)
end

function (==)( a::T ,b::Real) where T <: Tensor
    return false
end
function (==)(a::T,b::U) where {T <: Tensor, U <: SymbolicObject}
    return false
end
function (==)(a::T,b::U) where {T <: Tensor, U <: Tensor}
    return a.__pyobject__ == b.__pyobject__
end
function scalarIsEqual(a::U,b::T,metric::TensorHead) where {T <: Union{SymbolicObject,Tensor}, U <: Union{SymbolicObject,Tensor}}
    sca = get_scalars(a)
    scb = get_scalars(b)
    if 2 > length(sca) && 2 > length(scb)
        exa = canon_bp(contract_metric(scalar_exprs[sca[1]],metric))
        exb = canon_bp(contract_metric(scalar_exprs[scb[1]],metric))
        return exa == exb
    elseif length(sca) > 0 || length(scb) > 0
        error("Multiple scalars, not implemented yet.")
        return
    else
        return sympy.Eq(a,b) == True
    end
end

macro heads(IT::AbstractArray{TensorIndexType},xs...)
    hs = []
    for x in xs
        push!(hs,TensorHead(x,IT))
    end
    return hs
end

function canon_bp(s::Sym)
    sc = get_scalars(s)
    subs = Dict()
    if length(sc) > 0
        for ss in sc
            subs[ss] = canon_bp(scalar_exprs[ss])
        end
    end
    pyexp = s
    for k in keys(subs)
        if typeof(subs[k]) != TensMul
            pyexp = pyexp.subs(k,subs[k])
        end
    end
    return sympy_type_convert(pyexp+0)
end

function canon_bp(t::T) where T <: Tensor
    sc = get_scalars(t)
    subs = Dict()
    if length(sc) > 0
        for ss in sc
            subs[ss] = canon_bp(scalar_exprs[ss])
        end
    end
    pyexp = tensor.canon_bp(t)
    for k in keys(subs)
        if typeof(subs[k]) != TensMul
            pyexp = pyexp.subs(k,subs[k])
        end
    end
    return sympy_type_convert(pyexp+0)
end
function contract_metric(s::Sym,metric::TensorHead)
    sc = get_scalars(s)
    subs = Dict()
    if length(sc) > 0
        for ss in sc
            subs[ss] = contract_metric(scalar_exprs[ss],metric)
        end
    end
    pyexp = tensor.contract_metric(s,metric)
    for k in keys(subs)
        if typeof(subs[k]) != TensMul
            pyexp = pyexp.subs(k,subs[k])
        end
    end
    return sympy_type_convert(pyexp+0)
end
function contract_metric(t::T,metric::TensorHead) where T <: Tensor
    sc = get_scalars(t)
    subs = Dict()
    if length(sc) > 0
        for ss in sc
            subs[ss] = contract_metric(scalar_exprs[ss],metric)
        end
    end
    pyexp = tensor.contract_metric(t,metric)
    for k in keys(subs)
        if typeof(subs[k]) != TensMul
            pyexp = pyexp.subs(k,subs[k])
        end
    end
    return sympy_type_convert(pyexp+0)
end
contract_metric(x, metric::TensorHead) = x
canon_bp(x) = x

function factor(t::T) where T <: Tensor
    return sympy_type_convert(SymPy.factor(convert(Sym,t.__pyobject__)))
end
factor(x) = x
function simplify(t::T) where T <: Tensor
    return sympy_type_convert(SymPy.simplify(convert(Sym,t.__pyobject__)))
end

#function contract_delta(t::TensAdd, delta::TensorHead)
#    return sum([ contract_delta(ti,delta) for ti in terms(t) ])
#end
#function contract_delta(t::TensMul, delta::TensorHead)
#    
#end
