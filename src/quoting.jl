function Quote(ex::N) where {N <: Number}
    if ex isa Int || ex isa Float64 || ex isa Float32
        return ex
    else
        return convert(Expr, ex)
    end
end

# QuoteFn
function Quote(name::A,ex::Expr,vars=free_symbols(ex)) where {A <: AbstractString}
    return Expr(:function, Expr(:call, Symbol(name), map(Symbol,vars)...), Quote(ex))
end

# QuoteFnCSE
function Quote(name::A,cse_ex::Tuple{<:Array, Array{Sym,1}},vars::Array{Sym,1}) where {A <: AbstractString}
    quot = Expr(:block)
    quot.head = :function
    push!(quot.args, :($(Symbol(name))($(map(Symbol,vars)...) )))
    fnbody = Quote(cse_ex)
    push!(quot.args,fnbody)
    return quot
end

# QuoteCSE
function Quote(cse_ex::Tuple{T, Array{Sym,1}}) where T <: AbstractArray
    cse,ex = cse_ex
    ex=ex[1]
    fnbody=Expr(:block)
    for (k,v) in cse
        push!(fnbody.args, :($(Quote(k)) = $(Quote(v))) )
    end
    push!(fnbody.args,Quote(ex))
    return fnbody
end

#QuoteArrCSE
function Quote(cse_ex::Tuple{T,Array{U,1}}) where {T <: AbstractArray, U <: Array{Sym}}
    # assume we did simultaneous CSE on the array
    # cse is common list of substitutions
    # ex is array of cse'd expressions
    cse,ex = cse_ex
    fnbody=Expr(:block)
    for (k,v) in cse
        push!(fnbody.args, :($(Quote(k)) = $(Quote(v))) )
    end
    if length(ex) == 1
        ex = ex[1]
    end
    #push!(fnbody.args,:([$(Quote.(ex)...)]))
    push!(fnbody.args,Quote(ex).args[1])
    return fnbody
end

# QuoteArr
function Quote(ex::Array{Sym})
    fnbody=Expr(:block)
    sh = size(ex)
    push!(fnbody.args,:(reshape([$(Quote.(ex)...)],$sh)))
    # possibly improve this with alg at https://github.com/symengine/SymEngine.jl/pull/114
    return fnbody
end

# QuoteFnArr
function Quote(name::A,ex::Array{Sym}, vars=free_symbols(ex)) where {A <: AbstractString}
    Expr(:function, Expr(:call, Symbol(name), map(Symbol,vars)...), Quote(ex).args[1])
end

# QuoteFnArrCSE
function Quote(name::A, cse_ex::Tuple{<:AbstractArray,Array{<:Array{Sym},1}}, vars::Array{Sym,1}) where {A <: AbstractString}
    quot = Expr(:block)
    quot.head = :function
    push!(quot.args, :($(Symbol(name))($(map(Symbol,vars)...) )))
    fnbody = Quote(cse_ex)
    push!(quot.args,fnbody)
    return quot
end
