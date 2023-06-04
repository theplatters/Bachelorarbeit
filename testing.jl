include("wrapper.jl")
using .Hysteresis
using ForwardDiff
using LinearAlgebra
χ = 10.0;
h = mₚ = zeros(3);
A = diagm([5, 3, 1]);
b = zeros(3);
S(u) = u' * A * u + b' * u + 5;

U(m) = 0.5 * (m - b)' * inv(A) * (m - b) - 5;

norm(A)

x0 = [1.0, 10.0, 0.0];

prob = RestrainedProblem(χ, h, mₚ, U);

uprob = UnrestrainedProblem(χ,h,mₚ,S);

intf1 = Interface(uprob, x0);
inftf2 = Interface(prob,x0);

sol1 = solve(intf1,maxiter=100, tol=10^-10);

sol2 = solve(inftf2,maxiter = 100)

sol1.xk
sol2.xk

