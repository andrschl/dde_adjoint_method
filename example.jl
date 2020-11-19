cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate(); using Revise
using Flux, Zygote
using DifferentialEquations, DiffEqSensitivity
using DataStructures, LinearAlgebra

# define some constants
data_dim = 2
# lags = []
lags = Array(0.1:0.1:1)
ndelays = length(lags)
t = Array(5.0:0.2:20)
t_span = (t[1], t[end])

include("dde_adjoint.jl")

################################################################################
# Verify method using some basic neural network
u0 = ones(2)
model = Chain(Dense(data_dim*(1+ndelays),10, tanh), Dense(10,2))
p, re = Flux.destructure(model)
f=(u,p) -> re(p)(u)
t_span = (t[1], t[end])

function ndde_func(u, h, p, t)
    global lags
    ut = vcat(u, map(τ -> h(p,t-τ), lags)...)
    return f(ut, p)
end
function ndde_func!(du, u, h, p, t)
    global lags
    ut = vcat(u, map(τ -> h(p,t-τ), lags)...)
    du .= f(ut, p)
end

# calculate solution on t∈[0,10]
h0 = (p,t;idxs=nothing) -> ones(data_dim)
alg = MethodOfSteps(Tsit5())
sol = solve(DDEProblem(ndde_func!, u0, h0, (0,20), p=p), alg, u0=u0, p=p, dense=true)
t_span



# now calculate AD gradients on [5,10] (LSQ loss and zero data). The transition
# between history and solution should now be smooth enough
u5 = sol(5)
h5 = (p,t;idxs=nothing) -> sol(t)
prob = DDEProblem(ndde_func!, u5, h5, t_span, p=p)
ps = Flux.params(p)
@time begin
    gs = gradient(ps) do
        pred = Array(solve(prob, alg, u0=u5, p=p, saveat=t, sensealg = ReverseDiffAdjoint()))
        # pred = Array(solve(prob, alg, u0=u5, p=p, saveat=t, sensealg = ForwardDiffSensitivity()))
        sum(dot.(eachcol(pred),eachcol(pred)))
    end
end
gp = gs[p]
t
# custom interpolating adjoint sensitivity method
sol
loss = (x,y)->(x-y)'*(x-y)
@time dldp = interpolating_dde_adjoint(sol, f, p, t, loss, lags)
(dldp-gp)./dldp
