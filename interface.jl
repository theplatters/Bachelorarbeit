

using ForwardDiff, LinearAlgebra

J(r, ϕ, θ) =
    [[sin(θ) * cos(ϕ), r * cos(θ) * cos(ϕ), -r * sin(θ) * sin(ϕ)],
        [sin(θ) * sin(ϕ), r * cos(θ) * sin(ϕ), r * sin(θ) * cos(ϕ)],
        [cos(θ), -r * sin(θ), 0]]


struct RestrainedProblem
    χ::Float64
    h::Vector{Float64}
    mₚ::Vector{Float64}

    S::Function

    ∂S::Function
    ∂²S::Function

    obj::Function
    ∇obj::Function
    ∇²obj::Function

    objOnBall::Function
    ∇objOnBall::Function
    ∇²objOnBall::Function

end


function RestrainedProblem(χ, h, mₚ, S; kwags...)


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

    transformToEuklidean(ϕ, θ, r, h) = h + [r * sin(θ) * cos(ϕ), r * sin(θ) * sin(ϕ), r * cos(θ)]

    obj(u) = S(u) - u ⋅ mₚ
    ∇obj(u) = jac(u) - mₚ
    ∇²obj = hes
    objᵩ(x) = obj(transformToEuklidean(x..., χ, h))
    ∇objᵩ(x) = ForwardDiff.gradient(objᵩ, x)
    ∇²objᵩ(x) = ForwardDiff.hessian(objᵩ, x)
    RestrainedProblem(χ, h, mₚ, S, jac, hes, obj, ∇obj, ∇²obj, objᵩ, ∇objᵩ, ∇²objᵩ)
end


struct UnrestrainedProblem
    χ::Float64
    h::Vector{Float64}
    mₚ::Vector{Float64}

    U::Function


    ∂U::Function
    ∂²U::Function

    diffPart::Function
    ∇diffPart::Function

    obj::Function
    ∇obj::Function
    ∇²obj::Function

end

function UnrestrainedProblem(χ, h, mₚ, U; kwags...)
    println(kwags)

    if :jac ∉ keys(kwags)
        jac(x) = ForwardDiff.gradient(U, x)
    else
        jac = kwags[:jac]
    end


    if :hes ∉ keys(kwags)
        hes(x) = ForwardDiff.hessian(U, x)
    else
        hes = kwags[:hes]
    end

    diffPart(m) = U(m) - h ⋅ m 
    ∇diffPart(m) = jac(m) - h

    obj(m) = U(m) - h ⋅ m + χ*norm(m - mₚ)
    ∇obj(m) = ForwardDiff.gradient(U,m) - h + χ/norm(m - mₚ) * (m - mₚ)
    ∇²obj(m) = ForwardDiff.hessian(obj,m)



    UnrestrainedProblem(χ, h, mₚ, U, jac, hes,diffPart,∇diffPart,obj,∇obj,∇²obj)


end


mutable struct Interface{T}
    prob::T
    x0::Vector{Float64}
    xk::Vector{Float64}
    err::Float64
end

function Interface(prob::RestrainedProblem, x0)
    xk = x0
    err = Inf64

    Interface(prob, x0, xk, err)
end


function Interface(prob::UnrestrainedProblem,x0)
    xk = x0
    err = Inf64

    Interface(prob, x0, xk, err)
end


struct Solution
    prob
    xk::Vector{Float64}
    err::Float64
    convergent::Bool
end


