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

#struct TensScalar <: Tensor
#    var::Sym;
#    expr::Sym;
#    __pyobject__::PyCall.PyObject;
#end

struct TensMul <: Tensor
    __pyobject__::PyCall.PyObject
end
struct TensAdd <: Tensor
    __pyobject__::PyCall.PyObject
end
struct IndexedTensor <: Tensor
    __pyobject__::PyCall.PyObject
end

function SymPy._convert(::Val{:MutableDenseNDimArray}, x)
    sh = x.shape
    reshape(x.tolist(), sh)
end

function sympy_type_convert(pyexp)
    if typeof(pyexp) in  [Sym,PyObject]
        cname = pyexp.__class__.__name__
        if cname == "TensMul"
            return convert(TensMul,pyexp)
        elseif cname == "TensAdd"
            return convert(TensAdd,pyexp)
        elseif cname == "Tensor"
            return convert(IndexedTensor,pyexp)
        elseif cname == "TensorSymmetry"
            return convert(TensorSymmetry,pyexp)
        elseif cname == "MutableDenseMatrix"
            return convert(Array{Sym},pyexp)
        elseif cname == "MutableDenseNDimArray"
            return convert(Array{Sym},pyexp)
        elseif cname == "ImmutableDenseNDimArray"
            if typeof(pyexp) == Sym
                return pyexp.__pyobject__.tolist()
            else
                return pyexp.tolist()
            end
        elseif cname == "Zero"
            return 0
        elseif cname == "Half"
            return 1//2
        elseif cname == "One"
            return 1
        elseif cname == "Rational"
            return convert(Rational,pyexp)
        elseif cname == "Integer"
            return convert(Int,pyexp)
        elseif cname == "Float"
            return convert(Float64,pyexp)
        else
            return pyexp
        end
    else
        return pyexp
    end
end
