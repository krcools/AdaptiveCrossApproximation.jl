using AdaptiveCrossApproximation, Test

using LinearAlgebra
using CompScienceMeshes
using BEAST
using ParallelKMeans
using H2Trees

@testset "HMatrix: skipfars" begin

    m0 = meshsphere(radius=1.0, h=0.45)
    m1 = CompScienceMeshes.translate(m0, [0.0, 0.0, -10.0])
    m2 = CompScienceMeshes.translate(m0, [0.0, 0.0, +10.0])
    m = CompScienceMeshes.weld(m1, m2)

    X = raviartthomas(m)
    n = div(numfunctions(X), 2)

    a = Maxwell3D.singlelayer(; gamma=2.0);
    ttree = KMeansTree(X.pos, 2; minvalues=100)
    tree = BlockTree(ttree, ttree)

    function isnear(treea::H2Trees.BoundingBallTree, treeb::H2Trees.BoundingBallTree, nodea::Int, nodeb::Int)
        ths = H2Trees.radius(treea, nodea)
        shs = H2Trees.radius(treeb, nodeb)
        dist = norm(H2Trees.center(treea, nodea) - H2Trees.center(treeb, nodeb)) - (ths + shs)
        dist < 5 * alpha
    end

    A = HMatrix(a, X, X, tree;
        tol=1e-3,
        maxrank=40,
        isnear=AdaptiveCrossApproximation.isnear(),
        spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
        skipassemblefars=true
    )

    I1 = findall(getindex.(X.pos, 3) .> 0.0)
    I2 = findall(getindex.(X.pos, 3) .< 0.0)
    u = ones(2n)
    u[I1] .= 0
    v = A * u
    @test norm(v[I1]) ≈ 0 atol=1e-8
end
