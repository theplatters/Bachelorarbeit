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

    function RestrainedProblem(χ, h, mₚ, S, u0)
        jac(x) = ForwardDiff.jacobian(S, x)
        hes(x) = ForwardDiff.hessian(S, x)
        OptimProb(χ, h, mₚ, S, u0, jac, hes, false, false)

    end

    function RestrainedProblem(χ, h, mₚ, S, u0; kwags...)
        println(kwags)
        if :jac ∉ keys(kwags)
            jac(x) = ForwardDiff.jacobian(S,x)
            hes(x) = ForwardDiff.hessian(S,x)
            return OptimProb(χ, h, mₚ, S, u0, jac, hes, false, false)
        end

        
        jac = kwags[:jac]


        if :hes ∉ keys(kwags)
            hes(x) = ForwardDiff.hessian(S,x);
            print(hes)
            return OptimProb(χ, h, mₚ, S, u0, jac, hes, true, false)
        end

        hes = kwags[:hes]

        OptimProb(χ, h, mₚ, S, u0, kwags, hes, true, true)
    end
end

struct UnrestrainedProblem

    χ::Float64
    h::Vector{Float64}
    mₚ::Vector{Float64}


    U::Function

    m0::Vector{Float64}

    ∂U::Function
    ∂²U::Function

    hasObjectiveGradient::Bool = false
    hasObjectiveHessian::Bool = false
end

struct TwoProblems
    Restrained::RestrainedProblem
    Unrestrained::UnrestrainedProblem
end



χ = 1.0;
h = [1.0, 1.0, 1.0];
mₚ = h;
S(x) = x[1]^2 + x[3]^2 + 3 * x[2]^2
∂S(x) = [2 * x[1], 6 * x[2], 2 * x[3]]

u0 = m0 = [1.0, 1.0, 1.0]

U = S
∂U = ∂S

prob = RestrainedProblem(χ, h, mₚ, S, u0, jac = ∂S)

prob.∂²S([1.0, 1.99, 1.0])

testdict = Dict(:a => ∂S,:b => S)

testdict[:b]