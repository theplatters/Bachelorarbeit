module Solvers

include("interface.jl")
using LinearAlgebra, .HysteresisInterface

@enum Algorithms begin
    DEFAULT
    NEWTON
    PROXGRAD
end

function solve(intf :: Interface; algorithm :: Algorithms = DEFAULT, maxiter=1000, tol=10^-10)
    if typeof(intf.prob) == RestrainedProblem
        if algorithm == DEFAULT || algorithm == NEWTON
            sol = newton(intf,maxiter = maxiter,tol = tol)
            if(norm(sol.xk - intf.prob.h) > χ)
                
            end
            return sol
        end
    else

    end
end

function newton(interface; maxiter=1000, tol=10^-10)

    for i in 1:maxiter

        x = (interface.prob.∇²obj(interface.xk) \ -interface.prob.∇obj(interface.xk)) + interface.xk
        interface.err = norm(x - interface.xk)
        interface.xk = x

        if (interface.err < tol)
            return Solution(interface.prob, interface.xk, interface.err, true)
        end
    end
    print("Reached MaxIter, solution is maybe not convergent")
    return Solution(interface.prob, interface.xk, interface.err, false)
end



end