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

    obj::Function
    ∇obj::Function
    ∇²obj::Function

    objᵩ::Function
    ∇objᵩ::Function
    ∇²objᵩ::Function

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

    obj(u) = S - u ⋅ mₚ
    ∇obj(u) = jac(u) - mp
    ∇²obj = hes
    objᵩ(ϕ, θ) = obj(transformToEuklidean(ϕ, θ, χ, h))
    ∇objᵩ(ϕ, θ) = ∇obj(transformToEuklidean(ϕ, θ, χ, h))' * J(χ, ϕ, θ)
    ∇²objᵩ(ϕ, θ) = ForwardDiff.hessian((x) -> objᵩ(x...), [ϕ, θ])
    RestrainedProblem(χ, h, mₚ, S, u0, jac, hes, obj,∇obj,∇²obj,objᵩ,∇objᵩ,∇²objᵩ)
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

    xk::Vector{Float64}
    err::Float64
end

function Interface(prob::RestrainedProblem)
    xk = prob.x0
    err = Inf64

    Interface(prob, xk, err)
end


function Interface(prob::UnrestrainedProblem)
    xk = prob.x0
    err = Inf64

    Interface(prob, xk, err)
end


struct Solution
    prob
    xk::Vector{Float64}
    err::Float64
    convergent::Bool
end


end
