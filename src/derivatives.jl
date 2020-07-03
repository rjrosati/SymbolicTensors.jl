# make diff work on tensors
diff(A::Real, B::T) where {T <: Tensor} = 0
function diff(A::T, B::U) where {T <: Tensor,U <: Tensor}
    if typeof(B) != IndexedTensor
        error("Derivative wrt $(typeof(B)) objects is not supported.")
    elseif length(B.free_indices) > 1
        error("Can only differentiate wrt one free index.")
    end
    if typeof(A) == IndexedTensor
        if A.head == B.head
            i = B.free_indices[1]
            if i in A.indices
                error("What do we do here?")
            else
                j = A.indices[1]
                return A.head.index_types[1].metric(j,-i)
            end
        end
    elseif typeof(A) == TensAdd
        terms = SymbolicTensors.terms(A)
        l = length(terms)
        return sum([diff(t,B) for t in terms])
    elseif typeof(A) == TensMul
        terms = SymbolicTensors.terms(A)
        l = length(terms)
        if l > 1
            ans = []
            for (i,t) in enumerate(terms)
                if typeof(t) == TensMul
                    push!(ans, t.coeff * prod(terms[1:l .!= i])*diff(t.nocoeff,B))
                else
                    push!(ans, prod(terms[1:l .!= i])*diff(t,B))
                end
            end
            return sum(ans)
        else
            return A.coeff * diff(A.nocoeff,B)
        end
    else
        error("Differentiation of $(typeof(A)) not implemented")
    end
end
