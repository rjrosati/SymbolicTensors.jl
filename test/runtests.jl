using SymbolicTensors
using SymPy
using Test

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
        @test diff(field.delta(i,j),A(k)) == 0
        #@test gg(-i,-j,-k) ==
        #h = canon_bp(1//2*(gg(-i,-j,-k) + gg(-i,-k,-j) - gg(-j,-k,-i) ))
        #GG = TensorHead("GG",[field,field,field])
        #h = canon_bp(1//2*(GG(-i,-j,-k) + GG(-i,-k,-j) - GG(-j,-k,-i) ))
        #@test h(i,-j,-k) == h(i,-k,-j)
    end

    @testset "errors" begin
    end
end
