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
function time_window_population_cv(sol, window_size; step_size = 0.1)

    fwm = sol.prob.f.sys
    spp = variables(fwm, type = SPECIES_VARIABLE)

    cvs = zeros(Float64, length(sol.t))

    # Calculate CoV for each species
    for (i, t) in enumerate(sol.t)
 
        t1 = max(t - window_size/2.0, sol.t[1])
        t2 = min(t + window_size/2.0, sol.t[end])

        m = [(mean ∘ sol)(t1:t2, idxs = sp) for sp in spp]
        v = [(var  ∘ sol)(t1:t2, idxs = sp) for sp in spp]

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
function time_window_community_cv(sol, window_size)

    fwm = sol.prob.f.sys
    spp = variables(fwm, type = SPECIES_VARIABLE)

    cvs = zeros(Float64, length(sol.t))

    for (i, t) in enumerate(sol.t)

        t1 = max(t - window_size/2.0, sol.t[1])
        t2 = min(t + window_size/2.0, sol.t[end])

        community_total = sum([sol(t1:t2, idxs = sp) for sp in spp])

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

function jacobian_interaction_strength(sol)



end