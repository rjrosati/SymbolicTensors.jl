function replace_with_arrays(t::T, d::Dict) where {T <: SymbolicObject}
    sc = get_scalars(t)
    pydict = Dict([ k.__pyobject__ => v for (k,v) in d])
    pyexp = t.__pyobject__
    for s in sc
        pyexp = pyexp.subs(s,replace_with_arrays(scalar_exprs[s],d))
    end
    return sympy_type_convert(pyexp)
end
function replace_with_arrays(t::T, d::Dict) where {T <: Tensor}
    sc = get_scalars(t)
    pydict = Dict([ k.__pyobject__ => v for (k,v) in d])
    pyexp = t.__pyobject__
    for s in sc
        pyexp = pyexp.subs(s,replace_with_arrays(scalar_exprs[s],d))
    end
    pyexp = pyexp.replace_with_arrays(pydict)
    return sympy_type_convert(pyexp)
end

#function replace_with_arrays(t::T, d::Dict,TIT::TensorIndexType) where {T <: Tensor}
#    sc = get_scalars(t)
#    for s in sc
#        d[s] = replace_with_arrays(scalar_exprs[s],d)
#    end
#    heads = [ k.head  for (k,v) in d if typeof(k)==IndexedTensor ]
#    headkeys = Dict(k.head=> k  for (k,v) in d if typeof(k)==IndexedTensor )
#    ans = []
#    for t in terms(t)
#        if typeof(t) == IndexedTensor
#            if t.head in heads
#                if headkeys[t.head].is_up == t.is_up
#                    push!(ans,heads[t.head])
#                elseif TIT.metric in heads
#                    if all( x.is_up for headkeys[t.head].indices
#                        # if indices don't match, use metric to convert them
#                        push!(ans,d[headkeys[TIT.metric]]*)
#
#                else
#                    error("Couldn't replace $t, only had $(headkeys[t.head]) and no metric to convert them")
#                end
#            elseif t.head == TIT.delta
#                push!(ans,Matrix{I,dim,dim})
#            end
#        elseif typeof(t) == Sym
#            for s in sc
#                t = subs(t,s => d[s])
#            end
#            push!(ans,t)
#        else
#            push!(replace_with_arrays(t,d))
#        end
#    end
#    return prod(ans)
#end
