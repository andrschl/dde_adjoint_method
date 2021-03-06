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
returns discontuity times, which we add to constant_lags in order to account
for discontinuities in the adjoint state.
"""
#TODO: For discontinuities between initial history and solution it would be better to manually restart solver
function get_disc_points(t0, past_times, lags)
    taus = []
    append!(taus, lags)
    append!(taus, vcat(map(past_t -> relu.(past_t-t0.+lags), reverse(past_times))...))
    taus = setdiff(union(taus), [0])
    # remove duplicates and sort
    o = Lt((x,y) -> x < y - 1e-3)
    taus = SortedSet(taus, o)
    disc_points = []
    for i in taus
        push!(disc_points, i)
    end
    return disc_points
end
"""
returns array of relevant past sample times in t
"""
function get_past_times(t, t0_index, lags)
    τ_max = maximum(lags)
    t0 = t[t0_index]
    past_times = setdiff(map(t -> (t0-t>0)&&(t0-t<=τ_max) ? t : nothing, t), [nothing])
    return past_times
end
"""
returns upper bound of current t-interval which is the key for history dictionary TODO: double check this
"""
function get_key(t_current, t0, past_times)
    if t_current > t0
        return "nothing"
    else
        all_times = vcat(past_times, [t0])
        return string(Float64(all_times[searchsortedfirst(all_times, t_current)]))
    end
end
"""
Struct to describe initial history of adjoint state (which is discontinuous)
"""
mutable struct AdHistory
    h_dict::Dict{String, Function}
    t0::AbstractFloat
    past_times::AbstractArray
end
AdHistory(t0, past_times, key, value) = AdHistory(Dict("nothing"=>t->zeros(data_dim), key=>value), t0, past_times)
a = AdHistory(1.1, [], string(1.1), x->x)
(m::AdHistory)(t) = begin
    key = get_key(t, m.t0, m.past_times)
    return m.h_dict[key](t)
end
function interpolating_dde_adjoint(sol, f, p, t, l, lags; dl=nothing, data=zeros(data_dim, length(t)))
    if dl==nothing
        dl = (x,x_true)->gradient(y->l(y,x_true), x)[1]
    end
    alg = MethodOfSteps(Tsit5())
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
    dldp = zero(p)
    a0 = dl(sol(t[end]), data[:,1])
    h_a = AdHistory(ξ[1], [], string(ξ[1]), ξ -> ξ==0 ? a0 : zero(a0))
    past_times = []
    i = 2
    for ξ0 in ξ[1:end-1]
        ξ1 = ξ[i]
        ξ_span = (ξ0, ξ1)
        b0 = zero(p)
        s0 = vcat(a0, b0)
        if !isempty(lags)
            disc_points = get_disc_points(ξ0, past_times, lags)
        else
            disc_points = []
        end
        ad_prob = DDEProblem(adjoint_dde_func!, s0, (p,t;idxs=nothing)->h_a(t), ξ_span, p=p, constant_lags=disc_points)
        a_sol = solve(ad_prob, alg, u0=s0, p=p, dense=true)
        s1 = a_sol.u[end]

        # update history
        if !isempty(lags)
            push!(h_a.h_dict, string(ξ1) => ξ -> a_sol(ξ,idxs=1:data_dim))
            h_a.t0 = ξ1
            prev_past_times = past_times
            past_times = get_past_times(ξ, i, lags)
            h_a.past_times = past_times
            # delete old stuff in history dict
            for time in setdiff(prev_past_times, past_times)
                delete!(h_a.h_dict, string(Float64(time)))
            end
        end

        # update dldp and a0
        dldp += s1[data_dim+1:end]
        a1 = s1[1:data_dim]
        a0 =  a1 + dl(sol(T - ξ1), data[:, i]) # apply jump discontinuity
        i += 1
        GC.gc()
    end
    return dldp
end
