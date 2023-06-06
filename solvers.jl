
include("interface.jl")

export solve, newton, Algorithms
using LinearAlgebra

transformToEuklidean(ϕ, θ, r, h) = h + [r * sin(θ) * cos(ϕ), r * sin(θ) * sin(ϕ), r * cos(θ)]

transformToRadial(x, m) = [atan(x[2] - m[2], x[1] - m[1]), acos((x[3] - m[3]) / norm(x - m))]


function solve(intf::Interface{T}; maxiter=1000, tol=10^-10, algorithm=:default) where {T}
    if T == RestrainedProblem
        return newton(intf, maxiter=maxiter, tol=tol)
    else
        if algorithm == :default
            return proxGrad(intf, maxiter=maxiter, tol=tol)
        elseif algorithm == :quasiNewton
            return quasiNewton(intf, maxiter=maxiter, tol=tol)
        end
    end
end

function newton(g, H, x0; maxiter=1000, tol=1.e-10)
    xk = x0
    for i in 1:maxiter
        xk = (H(xk) \ -g(xk)) + xk
        err = norm(g(xk))

        if (err < tol)
            return xk, err, true, i
        end

    end
    print("Reached MaxIter, solution is maybe not convergent")
    return xk, err, false, maxiter
end


function newton(interface; maxiter=1000, tol=1.e-10)
    xk, err, conv, iter = newton(interface.prob.∇obj, interface.prob.∇²obj, interface.x0, maxiter=maxiter, tol=tol)
    if (norm(xk - interface.prob.h) ≤ interface.prob.χ) #Check if point is on Sphere
        return Solution(interface.prob, xk, err, conv, iter)
    else
        xk_centered = xk - interface.prob.h
        x0_euk = (interface.prob.χ / norm(xk_centered) * xk_centered) + interface.prob.h #Project solution onto sphere
        x0 = transformToRadial(x0_euk, interface.prob.h) #transform to radial coordinates
        xk, err, conv, iter = newton(interface.prob.∇objOnBall, interface.prob.∇²objOnBall, x0, maxiter=maxiter, tol=tol)
        xk_euk = transformToEuklidean(xk..., interface.prob.χ, interface.prob.h)
        return Solution(interface.prob, xk_euk, err, conv, iter)
    end
end

function proxOfNorm(x, λ, mp)
    ((1 - λ / max(norm(x - mp), λ)) * (x - mp)) + mp
end

function proxGrad(intf; maxiter=1000, tol=1.e-10)
    s = 0.1
    η = 1.4
    Lk = s
    err = 0.0
    f(xk) = intf.prob.U(xk) - xk ⋅ intf.prob.h
    ∂f(xk) = intf.prob.∂U(xk) - intf.prob.h
    T(∂f, Lk, xk) = proxOfNorm((xk .- 1 / Lk * ∂f(xk)), 1 / Lk * intf.prob.χ, intf.prob.mₚ)

    for i ∈ 1:maxiter
        while f(T(∂f, Lk, intf.xk)) > f(intf.xk) + dot(∂f(intf.xk), (T(∂f, Lk, intf.xk) - intf.xk)) + Lk / 2 * norm(T(∂f, Lk, intf.xk) - intf.xk)^2
            Lk = Lk * η
        end
        intf.xk = T(∂f, Lk, intf.xk)

        if intf.xk ≈ intf.prob.mₚ
            if norm(∂f(intf.xk)) ≤ intf.prob.χ
                err = norm(intf.xk - intf.prob.mₚ)
                return Solution(intf.prob, intf.xk, err, true, i)
            end
        else
            err = norm(intf.prob.∇obj(intf.xk))
            if (err ≤ tol)
                return Solution(intf.prob, intf.xk, err, true, i)
            end
        end
    end
    return Solution(intf.prob, intf.xk, err, false, maxiter)
end

function quasiNewton(intf; maxiter=1000, tol=1.e-10)
    err = 0.0
    for i ∈ 1:maxiter
        if intf.xk ≈ intf.prob.mₚ

            println(norm(intf.xk - intf.prob.mₚ))
            if intf.∂U(xk) + intf.prob.hᵣ ≤ χ
                return Solution(intf.prob, intf.xk, 0, true, i)
            else
                intf.xk = intf.xk + χ / norm(intf.xk) * intf.xk

            end
        else
            intf.xk = -0.9(intf.prob.∇²obj(intf.xk) \ intf.prob.∇obj(intf.xk)) + intf.xk
            err = norm(intf.prob.∇obj(intf.xk))

            if (err < tol)
                return Solution(intf.prob, intf.xk, err, true, i)
            end
        end

    end
    return Solution(intf.prob, intf.xk, err, false)
end