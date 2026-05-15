using LinearAlgebra
using CompScienceMeshes
using BEAST
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using Krylov
using PlotlyJS

struct BellCurveSingleLayerIOP{T} <: BEAST.IntegralOperator
    width::T
end

function (igd::BEAST.Integrand{<:BellCurveSingleLayerIOP})(p,q,f,g)
    α = igd.operator.width

    x = CompScienceMeshes.cartesian(p)
    y = CompScienceMeshes.cartesian(q)
    R = LinearAlgebra.norm(x-y)
    G = 4 * exp(-(R/α)^2) / (4 * π * R) / (sqrt(π) * α)

    BEAST._integrands(f,g) do fi,gj
        dot(fi.value, gj.value) * G
    end
end

function BEAST.scalartype(::BellCurveSingleLayerIOP{T}) where {T<:Real}
    T
end

function isnear(treea::H2Trees.BoundingBallTree, treeb::H2Trees.BoundingBallTree, nodea::Int, nodeb::Int)
    ths = H2Trees.radius(treea, nodea)
    shs = H2Trees.radius(treeb, nodeb)
    dist = norm(H2Trees.center(treea, nodea) - H2Trees.center(treeb, nodeb)) - (ths + shs)
    dist < 4 * alpha
end

Γ = meshsphere(1.0, 0.08);
X = raviartthomas(Γ);
@show numfunctions(X)

alpha = 0.1
op = BellCurveSingleLayerIOP(alpha)

ttree = KMeansTree(X.pos, 2; minvalues=100)
tree = BlockTree(ttree, ttree)

@time 𝐓 = HMatrix(op, X, X, tree;
    tol=1e-4,
    maxrank=40,
    isnear=isnear,
    spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
    skipassemblefars=true
)

AdaptiveCrossApproximation.storage(𝐓)
u = rand(ComplexF64, size(𝐓,2))
@time v = 𝐓 * u;

Q = rand(ComplexF64, size(𝐓))
@time w = Q * u;