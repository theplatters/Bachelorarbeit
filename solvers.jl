module Solvers

include("interface.jl")
using LinearAlgebra

function newton(interface; maxiter=1000, tol=10^-10)

    for i in 1:maxiter

        x = (interface.prob.∂²S(interface.xk) \ interface.prob.∂S(interface.xk)) + interface.xk
        interface.err = norm(x - interface.xk)
        interface.xk = x

        if (interface.err < tol)
            return interface
        end
    end
    print("Reached MaxIter, solution is maybe not convergent")
    return interface
end
end