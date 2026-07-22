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

    tls = (trophic_levels ∘ rescale_network)(web)
    return (maximum ∘ values)(tls) - (minimum ∘ values)(tls)
end

function maximum_trophic_level(web)::Float64

    tls = (trophic_levels ∘ rescale_network)(web)
    return (maximum ∘ values)(tls)
end

"""
    cascade_timespan(cascade::Vector{Tuple{Float64, Vector{Symbol}}}})

Returns the total timespan of an extinction cascade.
"""
function cascade_timespan(cascade::Vector{Tuple{Float64, Vector{Symbol}}})::Float64

    times = first.(cascade)

    if isempty(times)

        return 0 
    end

    return maximum(times)
end

"""
    cascade_timespan(cascade::Vector{Tuple{Float64, Vector{Symbol}}})

Returns the mean time to extinction after a primary extinction
"""
function mean_extinction_time(cascade::Vector{Tuple{Float64, Vector{Symbol}}})::Float64

    times = first.(cascade)

    if isempty(times)

        return 0 
    end

    return mean(times .- minimum(times))
end

"""
    median_interaction_strength(sol)

Calculates the median interaction over time in the community.

Interaction strength, here, is measured as preference parameters (α) of a
consumer species on it's resources.
"""
function median_interaction_strength(sol::ODESolution, sp::Symbol)::Float64

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

function median_interaction_strength(sol::ODESolution)::Float64

    fwm = sol.prob.f.sys
    out = Vector{Float64}(undef, length(sol.t))

    median_alphas = [median_interaction_strength(sol, sp) for sp in species(fwm)]

    for i in eachindex(sol.t)

        out[i] = mean([x[i] for x in median_alphas])
    end

    return out 
end

"""
    leading_eigenvalue(sol::ODESolution, t::Float64)::ComplexF64

Returns the leading (complex) eigenvalue of the system at time `t`
"""
function leading_eigenvalue(sol::ODESolution, t::Float64)::ComplexF64

    return (last ∘ eigvals)(jacobian(sol, t), sortby = abs)
end

"""
    richness(sol)

Returns a timeseries of species richness over time from an `ODESolution`
"""
function HigherOrderFoodwebs.richness(sol::ODESolution)::Vector{Int64}

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

    cascade = filter(t -> (t[1] > t1) & (t[1] < t2), secondary_extinctions)

    if isempty(cascade)

        return typeof(cascade)() 
    end

    for (i, extinction) in enumerate(cascade)

        cascade[i] = (extinction[1] - t1, extinction[2])
    end

    return cascade
end

function get_foodwebmodel(sol)

    fwm = sol.prob.f.sys
    @assert fwm isa FoodwebModel

    return fwm
end

"""
    jacobian(sol::ODESolution, t::Float64)::Matrix{Float64}

Returns the jacobian of the system at time `t` obtained via finite differencing.
Out-of-place version.
"""
function jacobian(sol::ODESolution, t::Float64)::Matrix{Float64}

    fwm = get_foodwebmodel(sol)
    jac_du = zeros(Float64, 
        (length ∘ variables)(fwm.vars), 
        (length ∘ variables)(fwm.vars)
    )
    jacobian!(jac_du, sol, t) 

    return jac_du
end

"""
    jacobian(sol::ODESolution, t::Float64)::Matrix{Float64}

Returns the jacobian of the system at time `t` obtained via finite differencing.
In-place version.
"""
function jacobian!(
    jac_du::Matrix{Float64}, 
    sol::ODESolution, 
    t::Float64
    )::Nothing

    fwm = get_foodwebmodel(sol)
    ps = sol.prob.p
    u  = sol(t, idxs = get_index(fwm.vars, variables(fwm.vars)))
    f(du, u) = sol.prob.f.f(du, u, ps, t)

    du = zeros(Float64, length(u))
    
    FiniteDiff.finite_difference_jacobian!(jac_du, f, u)

    return nothing 
end

function neighbouring_species(web, sp, r = 1)::Set{Symbol}

    if r == 1

        return neighbors(web, sp)

    else
        
        return union(
            neighbouring_species.(Ref(web), neighbors(web, sp), Ref(r - 1))...
        )
    end
end

function extract_neighbourhood(web, sp, r = 1)

    spp = (collect ∘ neighbouring_species)(web, sp, r)
    m   = spzeros(eltype(web.edges), length(spp), length(spp))

    for (i, subj) in enumerate(spp)
        for (j, obj) in enumerate(spp)

            m[i,j] = web[subj, obj]
        end
    end

    return SpeciesInteractionNetwork(Unipartite(spp), typeof(web.edges)(m))
end