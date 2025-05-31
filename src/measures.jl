"""
    median_interaction_strength(sol, t)

Calculates the median interaction strength in the community at time `t`.

Interaction strength, here, is measured as preference parameters (α) of a
consumer species on it's resources.
"""
function median_interaction_strength(sol, t)

    fwm = sol.prob.f.sys
    a_norms = Vector{Float64}()

    for intx in interactions(fwm)

        r = fwm.dynamic_rules[intx]
        vs = r.vars
        as = filter(x -> variable_type(fwm.vars, x) == TRAIT_VARIABLE, vs)
    
        if isempty(as)

            continue
        end

        focal = as[1]

        focal_val = sol(t, idxs = focal)
        total_val = sum([sol(t, idxs = x) for x in as])

        norm = focal_val / total_val
        push!(a_norms, norm)
    end

    return median(a_norms)
end

"""
    time_window_population_cv(sol, window_size)

Calculate a time series of community-averaged population coefficient of variation 
using a rolling time-window of `window_size`. The average at each time step, is of
only the species with non-zero mean population over the time window.
"""
function time_window_population_cv(sol, window_size; 
    step_size = 0.1, window_alignment = :CENTRE
)

    fwm = sol.prob.f.sys
    spp = variables(fwm, type = SPECIES_VARIABLE)

    cvs = zeros(Float64, length(sol.t))

    # Calculate CoV for each species
    for (i, t) in enumerate(sol.t)
 
        t1, t2 = time_window(sol, window_size, t, window_alignment)

        m = [(mean ∘ sol)(t1:step_size:t2, idxs = sp) for sp in spp]
        v = [(var  ∘ sol)(t1:step_size:t2, idxs = sp) for sp in spp]

        covs = v ./ m 

        cvs[i] = (mean ∘ filter)(!isnan, covs)
    end

    return cvs
end

"""
    time_window_community_cv(sol, window_size)

Calculate a time series of coefficient of variation of total community biomass
accross a rolling window of size `window_size`.
"""
function time_window_community_cv(sol, window_size; window_alignment = :CENTRE)

    fwm = sol.prob.f.sys
    spp = variables(fwm, type = SPECIES_VARIABLE)

    cvs = zeros(Float64, length(sol.t))

    for (i, t) in enumerate(sol.t)

        t1, t2 = time_window(sol, window_size, t, window_alignment)

        community_total = sum([sol(t1:0.001:t2, idxs = sp) for sp in spp])

        v = var(community_total)
        m = mean(community_total)

        if iszero(m)
            cvs[i] = 0.0
        else
            cvs[i] = v/m
        end    
    end

    return cvs
end

"""
    eigenstability(sol)

Calculate the real part of the leading eigenvalue of the jacobian matrix
at every point t along the time series.
"""
function eigenstability(sol; species_only = true)
 
    jac! = get_jacobian(sol) 
    fwm  = get_foodwebmodel(sol)

    n = (length ∘ variables)(fwm)
    s = richness(fwm)

    jval = zeros(n,n)

    param_vals = get_value.(Ref(fwm.params), variables(fwm.params))

    out = Vector{Float64}()
    sizehint!(out, length(sol.t))

    for t in sol.t

        jac!(jval, sol(t), param_vals, t)

        if species_only 
            e = (maxabs ∘ real)(eigen(jval).values)
        else 
            e = (maxabs ∘ real)(eigen(jval).values)
        end

        push!(out, e)
    end

    return out 
end

function HigherOrderFoodwebs.richness(sol::ODESolution)

    out = Vector{Int64}()
    sizehint!(out, length(sol.t))

    fwm = get_foodwebmodel(sol) 

    for t in sol.t

        s = count(x -> sol(t; idxs = x) > 0, species(fwm))
        push!(out, s)
    end

    return out
end

# ----------------- #
# Utility functions #
# ----------------- #

# TODO Add check that these actually exist. The sol isn't guaranteed to be
# coming from HigherOrderFoodwebs
function get_foodwebmodel(sol::ODESolution)

    return sol.prob.f.sys
end

function get_jacobian(sol::ODESolution)

    return sol.prob.f.jac
end

function time_window(sol, window_size, t, type = :CENTRE)

    @assert type in [:CENTRE, :LEFT, :RIGHT]
    @assert 0 <= t <= sol.t[end]
    @assert window_size <= sol.t[end] - sol.t[1]

    if type == :CENTRE

        t1 = max(t - window_size/2.0, sol.t[1])
        t2 = min(t + window_size/2.0, sol.t[end])

        return (t1, t2)
    end

    if type == :LEFT

        t1 = max(t - window_size, sol.t[1])
        t2 = min(t, sol.t[end])

        return (t1, t2)
    end

    if type == :RIGHT

        t1 = max(t, sol.t[1])
        t2 = min(t + window_size, sol.t[end])

        return (t1, t2)
    end

end

"""
    maxabs(x::Vector{Number})

Compare based on abs value but return the signed value.
"""
function maxabs(x::Vector{<:Number})

    max = x[1]

    for i in x

        if abs(i) > abs(max)

            max = i
        end
    end

    return max
end
