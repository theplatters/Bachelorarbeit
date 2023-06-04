include("wrapper.jl")
using .Hysteresis
using ForwardDiff
using LinearAlgebra
χ = 0.3;
h = mₚ = zeros(3);
A = diagm([5, 3, 1]);
b = [1.0 , 2.0 , 3.0];
S(u) = u' * A * u + b' * u + 5;

U(m) = 0.5 * (m - b)' * inv(A) * (m - b) - 5;

x0 = [1.0, 10.0, 0.0];

prob = RestrainedProblem(χ, h, mₚ, S);

uprob = UnrestrainedProblem(χ, h, mₚ, U);

intf1 = Interface(uprob, x0);
inftf2 = Interface(prob, x0);

sol1 = solve(intf1, maxiter=100, tol=10^-10);

sol2 = solve(inftf2, maxiter=100)

sol1.xk
sol2.xk

transformToRadial(x, m) = [atan(x[2] - m[2],x[1]-m[1]),acos((x[3] - m[3])/ norm(x-m))]
xk = transformToRadial(sol2.xk,h)
prob.∇objOnBall(xk)