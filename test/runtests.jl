using SymbolicTensors
using SymPy
using Test
using LinearAlgebra

@testset "SymbolicTensors.jl" begin
    @testset "types" begin
        field = TensorIndexType("field","f")
        @test typeof(field) == TensorIndexType
        @indices field i j
        @test typeof(i) == typeof(j) == TensorIndex
        A = TensorHead("A",[field])
        B = TensorHead("B",[field])
        @test typeof(A) == TensorHead
        @test typeof(A(i)) == IndexedTensor
        @test typeof(A(i)*A(-i)) == SymPy.Sym
        @test typeof(A(i) + B(i)) == TensAdd
        @test typeof(A(i) * B(j)) == TensMul
    end
    @testset "arithmetic" begin
        field = TensorIndexType("field","f")
        @indices field i j k l m n
        A = TensorHead("A",[field])
        B = TensorHead("B",[field])
        C = TensorHead("C",[field,field],get_tsymmetry.fully_symmetric(-2))
        @test A(i) + A(i) == 2*A(i)
        @test A(i) - A(i) == 0
        @test canon_bp(A(i)*A(j) + A(j)*A(i)) == 2*A(i)*A(j)
        @test canon_bp(C(-i,-j)*A(i)*A(j)) == 0
        # -- odd numbers of antisymmetric tensors contracted give zero
        @test canon_bp(C(i,-j)*C(j,-k)*C(k,-i)) == 0
        @test scalarIsEqual(A(i)*A(j)*field.metric(-i,-j),A(i)*A(-i),field.metric)
        @test scalarIsEqual(A(i)*B(j)*field.metric(-i,-j),A(i)*B(-i),field.metric)
        # should check properties of deltas, epsilons here too
        # -- should add some from xTensor
    end
    # -- should calculate some known Riemann scalars
    @testset "derivatives" begin
        field = TensorIndexType("field","f")
        @indices field i j k
        A = TensorHead("A",[field])
        @test diff(A(i),A(j)) == field.metric(i,-j)
        @test diff(A(-i),A(j)) == field.metric(-i,-j)
        @test diff(A(i),A(-j)) == field.metric(i,j)
        g = 1/(1+A(i)*A(-i))^2 * field.delta(-i,-j)
        gg = diff(g(-i,-j),A(k))
        @test diff(field.delta(-i,-j),A(k)) == 0
        # product rule w/ scalar chain rule
        @test scalarIsEqual( diff( A(i)*A(-i)*A(j), A(k)),
            A(i)*A(-i)*field.metric(j,-k) + 2*A(j)*A(-k) ,field.metric)
        # distribute over TensAdd
        @test canon_bp(diff( (A(i)*A(j) + A(j)*A(i))/2, A(k))) ==
            canon_bp(A(i)*field.metric(j,-k) + A(j)*field.metric(i,-k))
        #@test gg(-i,-j,-k) ==
        #h = canon_bp(1//2*(gg(-i,-j,-k) + gg(-i,-k,-j) - gg(-j,-k,-i) ))
        #GG = TensorHead("GG",[field,field,field])
        #h = canon_bp(1//2*(GG(-i,-j,-k) + GG(-i,-k,-j) - GG(-j,-k,-i) ))
        #@test h(i,-j,-k) == h(i,-k,-j)
    end

    @testset "errors" begin
    end

    @testset "replacements" begin
        field = TensorIndexType("field","f")
        @indices field i j k
        A = TensorHead("A",[field])
        field.set_metric(field.delta)
        @test replace_with_arrays(A(i)*A(j)*field.metric(-i,-j), Dict(field.delta(-i,-j)=>[1 0 ; 0 1], A(i)=>[0,4] )) == 16
    end

    @testset "quoting" begin
        vars = symbols("a b c d e f")
        a,b,c,d,e,f = vars
        function equalQuotes(q1,q2)
            if q1.head == :function
                q1 = q1.args[2]
                q2 = q2.args[2]
            end
            global a,b,c,d,e,f
            a,b,c,d,e,f = randn(6)
            return eval(q1) == eval(q2)
        end
        #Quote a scalar
        @test Quote(vars[1]) == convert(Expr,vars[1])
        cse_ex = sympy.cse(a^2 + 1/(1+a^2))
        @test equalQuotes(Quote(cse_ex),quote
              x0 = a ^ 2
              x0 + (1 + x0) ^ -1
        end)
        @test equalQuotes(Quote("hub",cse_ex,[a]),:(function hub(a)
              x0 = a ^ 2
              x0 + (1 + x0) ^ -1
        end))
        arr = [a^2,b^2+a^2,3c^2]
        cse_arr = sympy.cse(arr)
        @test equalQuotes(Quote(arr),quote
            reshape([a ^ 2, a ^ 2 + b ^ 2, __prod__(3, c ^ 2)], (3,))
        end)
        @test equalQuotes(Quote(cse_arr),quote
            x0 = a ^ 2
            reshape([x0, x0 + b ^ 2, __prod__(3, c ^ 2)], (3, 1))
        end)
        @test equalQuotes(Quote("_A",cse_arr,[a,b,c]),:(function _A(a, b, c)
            x0 = a ^ 2
            reshape([x0, x0 + b ^ 2, __prod__(3, c ^ 2)], (3, 1))
        end))
    end
    @testset "Riemann scalar computations" begin
        field = TensorIndexType("field","f")
        @indices field i j k l m n o
        F = TensorHead("F",[field])
        δ = field.delta

        simp(x) = contract_metric(canon_bp(x),field.metric)

        α = symbols("a",positive=true)
        g = 6 * α * δ(-i,-j) / (1-F(i)*F(j)*δ(-i,-j))^2
        gg = diff(g(-i,-j),F(k))
        gg = simp(gg)

        h = (gg(-i,-j,-k) - gg(-j,-k,-i) + gg(-k,-i,-j))/2
        h =  h(-i,-j,-k)
        h = simp(h)

        dh = diff(h(-i,-j,-k),F(l))
        dh = simp(dh)

        hh = (field.metric(m,n)*h(-m,-i,-j))*h(-n,-k,-l)
        hh = simp(hh)

        Riemann = dh(-i,-l,-j,-k) - dh(-i,-k,-l,-j) + hh(-j,-k,-i,-l) - hh(-j,-l,-i,-k)
        Riemann = simp(Riemann)

        Ric = Riemann(-i,-j,-k,-l)*field.metric(i,k)
        Ric = simp(Ric)

        R = Ric(-i,-j)*field.metric(j,i)
        R = simp(R)

        @test get_ricci(F, h, field, g) == Ric
        @test get_ricci_scalar(F, h, field, g) == R

        ## turn into arrays
        vars = symbols("x y z",real=true)
        x,y,z = vars
        d = length(vars)
        repl = Dict{Any,Any}(F(m) => vars, field.delta(-m,-n) => Matrix{Int64}(I,d,d))
        garr = simplify(replace_with_arrays(g(-i,-j),repl))

        repl[field] = garr
        #repl[field.metric(m,-n)] = Matrix{Int64}(I,d,d)
        harr = factor.(replace_with_arrays(h,repl))
        dharr = factor.(replace_with_arrays(dh,repl))
        hharr = factor.(replace_with_arrays(hh,repl))
        Rarr = factor(replace_with_arrays(R,repl))

        testpt = Dict(x=>0.1,y=>-0.8,z=>0.05,α=>1)
        @test float(subs(harr[1,1,1],testpt)) ≈ float(subs(-12*x / (-1+x^2+y^2+z^2)^3,testpt))
        @test float(subs(dharr[1,1,1,1],testpt)) ≈ float(subs(12*(1+5*x^2 - y^2 - z^2) / (-1+x^2+y^2+z^2)^4,testpt))
        @test float(subs(Rarr,testpt)) ≈ -4

        h = get_christoffel(F,field,g)
        Riemann = get_riemann(F,h,field,g)
        Ric = Riemann(-i,-j,-k,-l)*field.metric(i,k)
        Ric = simp(Ric)

        R = Ric(-i,-j)*field.metric(j,i)
        R = simp(R)

        @test get_ricci(F, h, field, g) == Ric
        @test get_ricci_scalar(F, h, field, g) == R

        vars = symbols("x y z",real=true)
        x,y,z = vars
        d = length(vars)
        repl = Dict{Any,Any}(F(m) => vars, field.delta(-m,-n) => Matrix{Int64}(I,d,d))
        garr = simplify(replace_with_arrays(g(-i,-j),repl))

        repl[field] = garr
        #repl[field.metric(m,-n)] = Matrix{Int64}(I,d,d)
        Rarr = factor(replace_with_arrays(R,repl))
        @test float(subs(Rarr,testpt)) ≈ -4
    end
end
