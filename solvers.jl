
include("interface.jl")

export solve, newton, Algorithms
using LinearAlgebra

transformToEuklidean(ϕ, θ, r, h) = h + [r * sin(θ) * cos(ϕ), r * sin(θ) * sin(ϕ), r * cos(θ)]

@enum Algorithms begin
    DEFAULT
    NEWTON
    PROXGRAD
end

function solve(intf::Interface{T}; maxiter=1000, tol=10^-10) where {T}
    if T == RestrainedProblem
        sol = newton(intf, maxiter=maxiter, tol=tol)
    else
        return proxGrad(intf, maxiter=maxiter, tol=tol)
    end
end

function newton(g, H, x0; maxiter=1000, tol=1.e-10)
    xk = x0
    for i in 1:maxiter
        x = (H(xk) \ -g(xk)) + xk
        err = norm(x - xk)
        xk = x

        if (err < tol)
            return xk, err, true
        end

    end
    print("Reached MaxIter, solution is maybe not convergent")
    return xk, err, false
end


function newton(interface; maxiter=1000, tol=1.e-10)
    xk, err, conv = newton(interface.prob.∇obj, interface.prob.∇²obj, interface.x0, maxiter=maxiter, tol=tol)
    if (norm(xk - interface.prob.h) ≤ interface.prob.χ)
        return Solution(interface.prob, xk, err, conv)
    else
        xk, err, conv = newton(interface.prob.∇objOnBall, interface.prob.∇²objOnBall, [0.0, 0.0], maxiter=maxiter, tol=tol)
        xk_euk = transformToEuklidean(xk..., interface.prob.χ, interface.prob.h)
        return Solution(interface.prob, xk_euk, err, conv)
    end
end

function proxOfNorm(x, λ, mp)
    ((1 - λ / max(norm(x - mp), λ)) * (x - mp)) + mp
end

function proxGrad(intf; maxiter=1000, tol=1.e-10)
    Lk = 5.916070
    err = 0.0
    for i ∈ 1:maxiter
        x = proxOfNorm((intf.xk .- 1 / Lk * intf.prob.∂U(intf.xk)), 1 / Lk * intf.prob.χ, intf.prob.mₚ)
        println("Iteration : $i, xk = $(intf.xk)")
        err = norm(intf.xk - x)
        if (err ≤ tol)
            return Solution(intf.prob, x, err, true)
        end
        intf.xk = x
    end
    return Solution(intf.prob, intf.xk, err, false)
end

