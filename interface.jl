module InterfaceHysteris

    export RestrainedProblem,UnrestrainedProblem,TwoProblems

    using ForwardDiff

    using ForwardDiff

    struct RestrainedProblem
        χ::Float64
        h::Vector{Float64}
        mₚ::Vector{Float64}
    
        S::Function
    
        u0::Vector{Float64}
    
        ∂S::Function
        ∂²S::Function
    
        hasObjectiveGradient::Bool
        hasObjectiveHessian::Bool
    
    end
    
    function RestrainedProblem(χ, h, mₚ, S, u0)
        jac(x) = ForwardDiff.gradient(S, x)
        Hessian(x) = ForwardDiff.hessian(S, x)
        RestrainedProblem(χ, h, mₚ, S, u0, jac, Hessian, false, false)
    
    end
    
    function RestrainedProblem(χ, h, mₚ, S, u0; kwags...)
        println(kwags)
    
        if :jac ∉ keys(kwags)
            jac(x) = ForwardDiff.gradient(S, x)
            hes(x) = ForwardDiff.hessian(S, x)
            return RestrainedProblem(χ, h, mₚ, S, u0, jac, hes, false, false)
        end
    
    
        jac = kwags[:jac]
    
    
        if :hes ∉ keys(kwags)
            hes(x) = ForwardDiff.hessian(S, x)
            return RestrainedProblem(χ, h, mₚ, S, u0, jac, hes, true, false)
        end
    
        Hessian = kwags[:hes]
    
        RestrainedProblem(χ, h, mₚ, S, u0, jac, Hessian, true, true)
    end
    
    struct UnrestrainedProblem
        χ::Float64
        h::Vector{Float64}
        mₚ::Vector{Float64}
    
        S::Function
    
        m0::Vector{Float64}
    
        ∂U::Function
        ∂²U::Function
    
        hasObjectiveGradient::Bool
        hasObjectiveHessian::Bool
    
    end
    
    function UnrestrainedProblem(χ, h, mₚ, U, u0)
        jac(x) = ForwardDiff.gradient(S, x)
        Hessian(x) = ForwardDiff.hessian(S, x)
        UnrestrainedProblem(χ, h, mₚ, U, u0, jac, Hessian, false, false)
    
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
    end
    
    struct TwoProblems
        Restrained::RestrainedProblem
        Unrestrained::UnrestrainedProblem
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
    
    
    
end