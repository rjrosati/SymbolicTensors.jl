# make diff work on tensors
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
                return A.head.index_types[1].delta(j,-i)
            end
        end
    elseif typeof(A) == TensAdd
        nothing
    elseif typeof(A) == TensMul
        ans = 0
        terms = terms(A)
        l = length(terms)
        for (i,t) in enumerate(terms)
            ans += prod(terms[1:d .!= i])*diff(t,B)
        end
        return ans
    else
        error("Differentiation of $(typeof(A)) not implemented")
    end
end
