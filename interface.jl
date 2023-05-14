module HysteresisInterface

    export RestrainedProblem,UnrestrainedProblem,TwoProblems,Interface

using ForwardDiff

struct RestrainedProblem
    χ::Float64
    h::Vector{Float64}
    mₚ::Vector{Float64}

    struct RestrainedProblem
        χ::Float64
        h::Vector{Float64}
        mₚ::Vector{Float64}
    
        S::Function
    
        x0::Vector{Float64}
    
        ∂S::Function
        ∂²S::Function
    
        hasObjectiveGradient::Bool
        hasObjectiveHessian::Bool
    
    end
    
    function RestrainedProblem(χ, h, mₚ, S, u0)
        jac(x) = ForwardDiff.gradient(S, x)
        hes(x) = ForwardDiff.hessian(S, x)
        return RestrainedProblem(χ, h, mₚ, S, u0, jac, hes, false, false)
    end


    jac = kwags[:jac]


    if :hes ∉ keys(kwags)
        hes(x) = ForwardDiff.hessian(S, x)
        return RestrainedProblem(χ, h, mₚ, S, u0, jac, hes, true, false)
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


    jac = kwags[:jac]


    if :hes ∉ keys(kwags)
        hes(x) = ForwardDiff.hessian(U, x)
        return UnrestrainedProblem(χ, h, mₚ, U, u0, jac, hes, true, false)
    end

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
    
    function TwoProblems(χ, h, mₚ,m0; S, U, kwargs...)
        u0 = m0
    
        if :∂S ∉ keys(kwargs)
            res = RestrainedProblem(χ, h, mₚ, S,u0)
        elseif :∂²S ∉ keys(kwargs)
            res = RestrainedProblem(χ, h, mₚ, S,u0,kwargs[:∂S])
        else
            res = RestrainedProblem(χ, h, mₚ, S,u0,kwargs[:∂S],kwargs[:∂²S])
        end
    
        if :∂U ∉ keys(kwargs)
            unres = UnrestrainedProblem(χ, h, mₚ, U,u0)
        elseif :∂²U ∉ keys(kwargs)
            unres = UnrestrainedProblem(χ, h, mₚ, S,u0,kwargs[:∂U])
        else
            unres = UnrestrainedProblem(χ, h, mₚ, S,u0,kwargs[:∂U],kwargs[:∂²U])
        end
    
        TwoProblems(res,unres)
    end
    
    mutable struct Interface
        prob :: Union{RestrainedProblem,UnrestrainedProblem}
        xk :: Vector{Float64}
        err :: Float64
    end

    function Interface(prob)
        xk = prob.x0
        err = Inf64
        Interface(prob,xk,err)
    end
    
end
