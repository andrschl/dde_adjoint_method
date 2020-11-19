"""
vector jacobian product with respect to input
"""
function VJP_x(f, x, a, p)
    y, back = Zygote.pullback(x->f(x,p), x)
    back(a)[1]
end
"""
vector jacobian product with respect to parameters
"""
function VJP_p(f, x, a, p)
    y, back = Zygote.pullback(p->f(x,p), p)
    back(a)[1]
end
"""
returns current delay state vector (x(t),x(t-τ_1),...,x(t-τ_k))
"""
function get_delayed_states(sol, t, lags)
    vcat(map(t->sol(t), t .- vcat([0], lags))...)
end

"""
returns C^n Discontinuities for 1 <= n <= order (due to adjoint jumps and not discontinuous DDE!)
"""
function get_higher_order_disc(ξ, lags; order=2, tol=1e-4)
    # define set of candidates
    ξ0 = ξ[1]
    ξ1 = ξ[end]
    o = Lt((x,y) -> x < y - tol)
    first_order_disc = vcat(map(t -> map(τ -> τ+t>ξ0 && τ+t<ξ1 ? τ+t : 0.0, lags), ξ)...)
    n_order_disc = SortedSet(first_order_disc, o)
    for i in 2:order
        prev_order = collect(n_order_disc)
        union!(n_order_disc, vcat(map(t -> map(τ -> τ+t>ξ0 && τ+t<ξ1 ? τ+t : 0.0, lags), prev_order)...))
    end
    setdiff!(n_order_disc,  ξ)
    return collect(n_order_disc)
end

function interpolating_dde_adjoint(sol, f, p, t, l, lags; dl=nothing, data=zeros(data_dim, length(t)))
    if dl==nothing
        dl = (x,x_true)->gradient(y->l(y,x_true), x)[1]
    end
    # we need the functions fi: xi->f(x0,...,xi,...,xk)
    fs = []
    for i in 0:ndelays
        g = function (x, xt, p)
            y = vcat(xt[1:data_dim*i],x,xt[(i+1)*data_dim+1:end])
            return f(y, p)
        end
        push!(fs, g)
    end
    # reverse time adjoint ξ := T - t, a(ξ) := λ(T-ξ), y(ξ) := x(T-ξ)
    T = t[end]
    function adjoint_dde_func!(ds, s, h, p, ξ)
        a = s[1:data_dim]
        yξ = get_delayed_states(sol, T - ξ, lags)
        y = yξ[1:data_dim]
        g = (y,p) -> fs[1](y, yξ, p)
        da =  VJP_x(g, y, a, p)
        ds[data_dim+1:end] .= VJP_p(f, yξ, a, p)
        for i in 1:ndelays
            ai = h(nothing, ξ-lags[i], idxs=1:data_dim)
            yξ = get_delayed_states(sol, T - ξ + lags[i], lags)
            y = yξ[i*data_dim+1:(i+1)*data_dim]
            gi = (y,p) -> fs[i+1](y, yξ, p)
            da +=  VJP_x(gi, y, ai, p)
        end
        ds[1:data_dim] .= da
    end

    ξ = reverse(T .- t)
    ξ0 = ξ[1]
    ξ1 = ξ[end]
    ξ_span = (ξ0, ξ1)
    a0 = dl(sol(t[end]), data[:,1])
    h_a = (p,ξ;idxs=nothing) -> ξ==0 ? a0 : zero(a0)

    # callbacks:
    # callback for adjoint jumps
    a_jumps = ξ[2:end-1]
    a_jump_idx = 1
    affect1!(integrator) = begin
        a_jump_idx += 1
        integrator.u[1:data_dim] += dl(sol(T - integrator.t), data[:, a_jump_idx])
    end
    cb1 = PresetTimeCallback(a_jumps,affect1!, save_positions=(true,true))

    # callback for adjoint C¹,C² discontinuities
    higher_order_disc = get_higher_order_disc(ξ, lags)
    affect2!(integrator) = nothing
    cb2 = PresetTimeCallback(higher_order_disc,affect2!)

    cb = CallbackSet(cb1,cb2)

    # integrate adjoint equation
    b0 = zero(p)
    s0 = vcat(a0, b0)
    ad_prob = DDEProblem(adjoint_dde_func!, s0, h_a, ξ_span, p=p)
    alg = MethodOfSteps(Tsit5())
    a_sol = solve(ad_prob, alg, u0=s0, p=p, saveat=[ξ1], callback=cb)
    s1 = a_sol.u[end]

    dldp = s1[data_dim+1:end]
    a1 = s1[1:data_dim]
    a0 =  a1 + dl(sol(T - ξ1), data[:, end])

    return dldp
end
