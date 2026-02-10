# This file contains a bunch of functions used for extracting some useful value
# from ODE solutions. These are mostly used within the `process_solution`
# function when we set up `EnsembleProblem`. It takes a lot of space to store 
# entire solution for so many runs, so the output of these functions is all we
# store of a simulation.
#
# See the corresponding file in ./scripts to find which measures are recorded in
# which simulations.

"""
    cascade_trophic_range(web, extinctions::Vector{Symbol})

Returns Δ trophic level of the species involved in a trophic cascade.
Returns zero if there was no cascade (or a single species).
"""
function cascade_trophic_range(web, extinctions::Vector{Symbol})::Float64

    if length(extinctions) < 2
        return 0.0 
    end

    # SpeciesInteractionNetworks.jl treats Quantitative networks in pathfinding
    # in a way that produces insane trophic levels. Distance of an interactions
    # is taken to be the inverse of the weight. Thus when the weights are
    # potentially small numbers 1 >>, you can get crazy things happening, like
    # everyone has a nice and reasonable trophic level, and them BAM! One
    # species is at trophic level 305. The package docs suggest using
    # `normalize` on the networks, but I've found that's not sufficient.  I
    # added the function `rescale_network` to HigherOrderFoodwebs.jl to deal
    # with this, but I'm not super happy about it in general. 

    tls = distancetobase.((Ref ∘ rescale_network)(web), extinctions, mean)
    return maximum(tls) - minimum(tls)
end

function maximum_trophic_level(web)::Float64

    # SpeciesInteractionNetworks.jl treats Quantitative networks in pathfinding
    # in a way that produces insane trophic levels. Distance of an interactions
    # is taken to be the inverse of the weight. Thus when the weights are
    # potentially small numbers 1 >>, you can get crazy things happening, like
    # everyone has a nice and reasonable trophic level, and them BAM! One
    # species is at trophic level 305. The package docs suggest using
    # `normalize` on the networks, but I've found that's not sufficient.  I
    # added the function `rescale_network` to HigherOrderFoodwebs.jl to deal
    # with this, but I'm not super happy about it in general. 

    tls = distancetobase.((Ref ∘ rescale_network)(web), species(web), mean)
    return maximum(tls) 
end

"""
    cascade_timespan(
    secondary_extinctions::Vector{Tuple{Float64, Symbol}},
    t1::Float64, t2::Float64)

Returns the timespan of the extinction cascade taking place between times t1 and
t2. This will produce NaN if there wasn't a cascade. 
"""
function cascade_timespan(secondary_extinctions, t1, t2)::Float64

    cascade = filter(t -> (t[1] > t1) & (t[1] < t2), secondary_extinctions)
    times = first.(cascade)

    if isempty(times)

        return 0 
    end

    return maximum(times) - minimum(times)
end

function mean_extinction_time(secondary_extinctions, t1, t2)::Float64

    cascade = filter(t -> (t[1] > t1) & (t[1] < t2), secondary_extinctions)
    times = first.(cascade)

    if isempty(times)

        return 0 
    end

    return mean(times .- minimum(times))
end

"""
    median_interaction_strength(sol)

Calculates the median interaction over time in the community.
sol.prob.f.sys
Interaction strength, here, is measured as preference parameters (α) of a
consumer species on it's resources.
"""
function median_interaction_strength(sol, sp)::Float64

    fwm = sol.prob.f.sys

    # A single interaction/rule should have all the relevant alpha variables.
    f(x) = (subject(x) == sp) & (!isloop(x))
    index = findfirst(f, interactions(fwm))

    # The case of a producer, no interactions with resouces found.
    # The interaction strength and thus it's median will always be zero.
    if isnothing(index)

        return zeros(Float64, length(sol.t))
    end

    intx = interactions(fwm)[index] 
    rule = fwm.dynamic_rules[intx]
    vars = filter(x -> variable_type(fwm, x) == TRAIT_VARIABLE, rule.vars)

    # The case of a consumer with a single resource. It's median will always one.
    if isempty(vars)

        return ones(Float64, length(sol.t))
    end

    median_alpha = Vector{Float64}(undef, length(sol.t))

    for t in eachindex(sol.t)

        m = median([sol[v, t] for v in vars])
        median_alpha[t] = m
    end

    return median_alpha
end

function median_interaction_strength(sol)::Float64

    fwm = sol.prob.f.sys
    out = Vector{Float64}(undef, length(sol.t))

    median_alphas = [median_interaction_strength(sol, sp) for sp in species(fwm)]

    for i in eachindex(sol.t)

        out[i] = mean([x[i] for x in median_alphas])
    end

    return out 
end

"""
    time_window_population_cv(sol, window_size)

Calculate a time series of community-averaged population coefficient of variation 
using a rolling time-window of `window_size`. The average at each time step, is of
only the species with non-zero mean population over the time window.
"""
function time_window_population_cv(sol, window_size; 
    step_size = 0.1, window_alignment = :CENTRE
)::Float64

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
function time_window_community_cv(sol, window_size; window_alignment = :CENTRE)::Float64

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
function eigenstability(sol; species_only = true, sparseness = 1)::Float64

    error("Broken, Don't use THIS!")    

    jac! = get_jacobian(sol) 
    fwm  = get_foodwebmodel(sol)

    n = (length ∘ variables)(fwm)
    s = richness(fwm)

    jval = zeros(n,n)

    param_vals = get_value.(Ref(fwm.params), variables(fwm.params))

    out = zeros(Float64, length(sol.t))
    missing_points = (trues ∘ length)(sol.t)

    for i in eachindex(sol.t)

        if i % sparseness == 0 

            continue
        end

        t = sol.t[i]

        jac!(jval, sol(t), param_vals, t)

        if species_only 
            e = (maxabs ∘ real)(eigen(jval).values)
        else 
            e = (maxabs ∘ real)(eigen(jval).values)
        end

        out[i] = e
        missing_points[i] = false
    end

    return (out, missing_points) 
end

"""
    richness(sol)

Returns a timeseries of species richness over time from an `ODESolution`
"""
function HigherOrderFoodwebs.richness(sol)::Vector{Int64}

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

function secondary_extinctions_during_trial(secondary_extinctions, t1, t2)

    return filter(t -> (t[1] > t1) & (t[1] < t2), secondary_extinctions)
end

function get_foodwebmodel(sol)

    fwm = sol.prob.f.sys
    @assert fwm isa FoodwebModel

    return fwm
end

function get_jacobian(sol)

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