using LinearAlgebra
using CompScienceMeshes
using BEAST
using ParallelKMeans
using H2Trees
using AdaptiveCrossApproximation
using Krylov
using PlotlyJS

struct BellCurveSingleLayerIOP{T,C} <: BEAST.IntegralOperator
    width::T
    gamma::C
end

function (igd::BEAST.Integrand{<:BellCurveSingleLayerIOP})(p,q,f,g)
    α = igd.operator.width
    γ = igd.operator.gamma
    γ⁻¹ = 1 / γ

    x = CompScienceMeshes.cartesian(p)
    y = CompScienceMeshes.cartesian(q)
    R = LinearAlgebra.norm(x-y)
    G = exp(-(R/α)^2) * exp(-γ * R) / (4 * π * R) #* 4 / (sqrt(π) * α)

    BEAST._integrands(f,g) do fi,gj
        -γ * dot(fi.value, gj.value) * G - γ⁻¹ * dot(fi.divergence, gj.divergence) * G 
    end
end

function BEAST.scalartype(::BellCurveSingleLayerIOP{T,C}) where {T,C}
    promote_type(T, C)
end



Γ = meshsphere(1.0, 0.04);
X = raviartthomas(Γ);
Y = buffachristiansen(Γ);
@show numfunctions(X)
@show numfunctions(Y)

κ = 1.0

E = Maxwell3D.planewave(; direction=ẑ, polarization=x̂, wavenumber=κ)
e = (n × E) × n
rhs = BEAST.assemble(e, X)

a = Maxwell3D.singlelayer(wavenumber=κ);
d = BEAST.NCross();
b1 = Maxwell3D.singlelayer(wavenumber=κ);

A = BEAST.assemble(a,X,X; threading=:cellcoloring)
A⁻¹ = BEAST.GMRESSolver(A; abstol=1e-4, reltol=1e-4, maxiter=1000)
u0, ch0 = BEAST.solve(A⁻¹, rhs)

D = BEAST.assemble(d,X,Y; threading=:cellcoloring)
B1 = BEAST.assemble(b1,Y,Y; threading=:cellcoloring)

D⁻¹ = BEAST.GMRESSolver(D; reltol=1e-10, maxiter=1000, verbose=false);
D⁻ᵀ = BEAST.GMRESSolver(D'; reltol=1e-10, maxiter=1000, verbose=false);
P1 = D⁻ᵀ * B1 * D⁻¹
A⁻¹ = BEAST.GMRESSolver(A; reltol=1e-4, maxiter=1000, left_preconditioner=P1)
u1, ch1 = BEAST.solve(A⁻¹, rhs)

alpha = 0.1
b2 = BellCurveSingleLayerIOP(alpha, κ); 
@time B2 = BEAST.assemble(b2,Y,Y; threading=:cellcoloring)
P2 = D⁻ᵀ * B2 * D⁻¹
A⁻¹ = BEAST.GMRESSolver(A; reltol=1e-4, maxiter=1000, left_preconditioner=P2)
u2, ch2 = BEAST.solve(A⁻¹, rhs)

# error()

function isnear(treea::H2Trees.BoundingBallTree, treeb::H2Trees.BoundingBallTree, nodea::Int, nodeb::Int)
    ths = H2Trees.radius(treea, nodea)
    shs = H2Trees.radius(treeb, nodeb)
    dist = norm(H2Trees.center(treea, nodea) - H2Trees.center(treeb, nodeb)) - (ths + shs)
    dist < 4 * alpha
end

ttree = KMeansTree(X.pos, 2; minvalues=100)
tree = BlockTree(ttree, ttree)


@time B3 = HMatrix(b2, Y, Y, tree;
    tol=1e-4,
    maxrank=40,
    isnear=isnear,
    spaceordering=AdaptiveCrossApproximation.PreserveSpaceOrder(),
    skipassemblefars=true
)
P3 = D⁻ᵀ * B3 * D⁻¹
A⁻¹ = BEAST.GMRESSolver(A; reltol=1e-4, maxiter=1000, left_preconditioner=P3)
u3, ch3 = BEAST.solve(A⁻¹, rhs)

@show ch0 ch1 ch2 ch3