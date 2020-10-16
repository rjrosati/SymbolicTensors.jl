# make diff work on tensors
diff(A::Real, B::T) where {T <: Tensor} = 0

## SymPy overrides to zero out Ints and Floats

SymPy.diff(::Int,::Sym...) = 0
SymPy.diff(::Float64,::Sym...) = 0

function diff(A::IndexedTensor,B::IndexedTensor)
    if length(B.free_indices) > 1
        error("For now, can only differentiate wrt one free index.")
    end
    if A.head == B.head
        i = B.free_indices[1]
        if i in A.free_indices
            error("What do we do here?")
        else
            j = A.free_indices[1]
            return A.head.index_types[1].metric(j,-i)
        end
    elseif A.head == B.head.index_types[1].delta
        tit = B.head.index_types[1]
        i, j = A.free_indices
        @indices tit m n
        if i.is_up == false && j.is_up == false
            return 0
        elseif i.is_up == false && j.is_up
            return diff(tit.metric(j,m)*A.head(-i,-m),B)
        elseif i.is_up && j.is_up == false
            return diff(tit.metric(i,m)*A.head(-m,-j),B)
        else
            return diff(tit.metric(i,m)*tit.metric(j,n)*A.head(-m,-n),B)
        end
    elseif A.head.__pyobject__.name[end] == 'c'
        return 0
    else
        println("Head1:", A.head)
        println("Head2:", B.head)
        error("Heads differ in a way not yet implemented")
    end
end

function diff(A::TensAdd,B::IndexedTensor)
    if length(B.free_indices) > 1
        error("Can only differentiate wrt one free index.")
    end
    ts = terms(A)
    return sympy_type_convert(sum([diff(t,B) for t in ts]))
end

function diff(A::TensMul,B::IndexedTensor)
    if length(B.free_indices) > 1
        error("Can only differentiate wrt one free index.")
    end
    trms = terms(A)
    l = length(trms)
    ans = []
    for (i,t) in enumerate(trms)
        if typeof(t) == TensMul
            dcoeff = t.coeff
            sc = get_scalars(t.coeff)
            if length(sc) == 1
                s = sc[1]
                dcoeff = diff(dcoeff,s) * diff(scalar_exprs[s],B)
            elseif length(sc) > 0
                error("Multiple scalars not implemented")
            else
                dcoeff = diff(dcoeff,B)
            end
            other = l>1 ? prod(trms[1:l .!= i]) : 1
            dt = (dcoeff * t.nocoeff + t.coeff*diff(t.nocoeff,B))
            if dt != 0
                push!(ans, dt*other)
            end
        else
            dt = diff(t,B)
            if dt != 0
                push!(ans, dt*prod(trms[1:l .!= i]))
            end
        end
    end
    return sympy_type_convert(sum([a for a in ans if typeof(a) <: Tensor]))
end

function diff(A::Sym,B::IndexedTensor)
    if length(B.free_indices) > 1
        error("Can only differentiate wrt one free index.")
    end
    sc = get_scalars(A)
    repl = Dict()
    dA = Dict()
    if length(sc) > 0
        for s in sc
            repl[s] = diff(scalar_exprs[s],B)
            dA[s] = diff(A,s) * diff(scalar_exprs[s],B)
        end
        return sum(values(dA))
    else
        #error("Can't differentiate non-scalar Sym objects wrt a tensor")
        return 0
    end
end

function diff(A::T, B::U) where {T <: SymbolicObject,U <: Tensor}
    if typeof(B) != IndexedTensor
        error("Derivative wrt $(typeof(B)) objects is not supported.")
    else
        error("Differentiation of type $(typeof(A)) not implemented")
    end
end
