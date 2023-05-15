include("interface.jl")
include("solvers.jl")
using .HysteresisInterface
using .Solvers
using ForwardDiff
using LinearAlgebra
χ = 1.0
h = mₚ = [1.,1.,1.]
A = diagm([5,3,10]);
S(u) = u' * A * u + [1.,-10.,1.]'*u + 5

x0 = [1.,10.,0.]

prob = RestrainedProblem(χ,h,mₚ,S,x0)

intf = Interface(prob)

sol = Solvers.newton(intf,maxiter = 10000,tol = 10^-10);

dump(sol)


transformToEuklidean(ϕ, θ, r, h) = h + [r * sin(θ) * cos(ϕ), r * sin(θ) * sin(ϕ), r * cos(θ)]

J(r, ϕ, θ) =
    [sin(θ) * cos(ϕ) r * cos(θ) * cos(ϕ) -r * sin(θ) * sin(ϕ);
        sin(θ) * sin(ϕ) r * cos(θ) * sin(ϕ) r * sin(θ) * cos(ϕ);
        cos(θ)  -r * sin(θ) 0]