using .InterfaceHysteris
using .Solvers

χ = 1.0
h = mₚ = [1.,1.,1.]

S(u) = u' * u + [1.,1.,1.]'*u + 5

x0 = [0.,0.,0.]

prob = RestrainedProblem(χ,h,mₚ,S,x0)

intf = Interface(prob)

sol = Solvers.newton(intf,maxiter = 10000,tol = 10^-10);