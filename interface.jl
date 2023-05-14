module HysteresisInterface

export RestrainedProblem, UnrestrainedProblem, TwoProblems, Interface, Solution

using ForwardDiff

J(r, ϕ, θ) =
    [[sin(θ) * cos(ϕ), r * cos(θ) * cos(ϕ), -r * sin(θ) * sin(ϕ)],
        [sin(θ) * sin(ϕ), r * cos(θ) * sin(ϕ), r * sin(θ) * cos(ϕ)],
        [cos(θ), -r * sin(θ), 0]]

transformToEuklidean(ϕ, θ, r, h) = h + [r * sin(θ) * cos(ϕ), r * sin(θ) * sin(ϕ), r * cos(θ)]

struct RestrainedProblem
    χ::Float64
    h::Vector{Float64}
    mₚ::Vector{Float64}

    S::Function

    x0::Vector{Float64}

    ∂S::Function
    ∂²S::Function

    Sᵩ::Function
    ∂Sᵩ::Function
    ∂²Sᵩ::Function

end


function RestrainedProblem(χ, h, mₚ, S, u0; kwags...)


    if !haskey(kwags, :jac)
        jac(x) = ForwardDiff.gradient(S, x)
    else
        jac = kwags[:jac]
    end

    if !haskey(kwags, :hes)
        hes(x) = ForwardDiff.hessian(S, x)
    else
        hes = kwags[:hes]
    end

    Sᵩ(ϕ, θ) = S(transformToEuklidean(ϕ, θ, χ, h))
    ∂Sᵩ(ϕ, θ) = hes(transformToEuklidean(ϕ, θ, χ, h))' * J(χ, ϕ, θ)
    ∂²Sᵩ(ϕ, θ) = ForwardDiff.hessian((x) -> Sᵩ(x...), [ϕ, θ])
    RestrainedProblem(χ, h, mₚ, S, u0, jac, hes, Sᵩ, ∂Sᵩ, ∂²Sᵩ)
end


struct UnrestrainedProblem
    χ::Float64
    h::Vector{Float64}
    mₚ::Vector{Float64}

    S::Function

    x0::Vector{Float64}

    ∂U::Function
    ∂²U::Function

    hasObjectiveGradient::Bool
    hasObjectiveHessian::Bool

end

function UnrestrainedProblem(χ, h, mₚ, U, u0; kwags...)
    println(kwags)

    if :jac ∉ keys(kwags)
        jac(x) = ForwardDiff.gradient(U, x)
        hes(x) = ForwardDiff.hessian(U, x)
        return UnrestrainedProblem(χ, h, mₚ, U, u0, jac, hes, false, false)
    end


    jac = kwags[:jac]


    if :hes ∉ keys(kwags)
        hes(x) = ForwardDiff.hessian(U, x)
        return UnrestrainedProblem(χ, h, mₚ, U, u0, jac, hes, true, false)
    end

    Hessian = kwags[:hes]

    UnrestrainedProblem(χ, h, mₚ, U, u0, jac, Hessian, true, true)

    Hessian = kwags[:hes]

    UnrestrainedProblem(χ, h, mₚ, U, u0, jac, Hessian, true, true)
end

struct TwoProblems
    Restrained::RestrainedProblem
    Unrestrained::UnrestrainedProblem
end



function TwoProblems(χ, h, mₚ, m0; S, U, kwargs...)
    u0 = m0

    if :∂S ∉ keys(kwargs)
        res = RestrainedProblem(χ, h, mₚ, S, u0)
    elseif :∂²S ∉ keys(kwargs)
        res = RestrainedProblem(χ, h, mₚ, S, u0, kwargs[:∂S])
    else
        res = RestrainedProblem(χ, h, mₚ, S, u0, kwargs[:∂S], kwargs[:∂²S])
    end

    if :∂U ∉ keys(kwargs)
        unres = UnrestrainedProblem(χ, h, mₚ, U, u0)
    elseif :∂²U ∉ keys(kwargs)
        unres = UnrestrainedProblem(χ, h, mₚ, S, u0, kwargs[:∂U])
    else
        unres = UnrestrainedProblem(χ, h, mₚ, S, u0, kwargs[:∂U], kwargs[:∂²U])
    end

    TwoProblems(res, unres)
end

mutable struct Interface{T}
    prob::T
    objectiveFunction::Function
    ∇obj::Function
    ∇²obj
    xk::Vector{Float64}
    err::Float64
end

function Interface(prob::RestrainedProblem)
    xk = prob.x0
    err = Inf64
    objectiveFunction(u) = prob.S(u) - u ⋅ prob.mₚ
    ∇obj(u) = prob.∂S(u) - prob.mₚ
    ∇²obj = prob.∂²S
    Interface(prob, objectiveFunction, ∇obj, ∇²obj, xk, err)
end


function Interface(prob::UnrestrainedProblem)
    xk = prob.x0
    err = Inf64
    objectiveFunction(u) = prob.U(u) - u ⋅ prob.h + prob.χ * norm(u - prob.mₚ)
    ∇obj(u) = prob.∂U(u) - h + χ / norm(u - prob.mₚ) * (u - prob.mₚ)
    ∇²obj(u) = prob.∂²U(u)
    Interface(prob, objectiveFunction, ∇obj, ∇²obj, xk, err)
end


struct Solution
    prob
    xk::Vector{Float64}
    err::Float64
    convergent::Bool
end


end
