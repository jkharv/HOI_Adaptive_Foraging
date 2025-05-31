include("../src/HOI_Adaptive_Foraging.jl")

using .HOI_Adaptive_Foraging

using OrdinaryDiffEqTsit5
using OrdinaryDiffEqRosenbrock

using SpeciesInteractionNetworks
using HigherOrderFoodwebs

using Symbolics
using Random
using LinearAlgebra
using Statistics
using Distributions
using DataFrames
using StatsBase
using CSV
using ADTypes

import WGLMakie
include("my_model.jl")

"""
    extra_save_points(times)

Returns a `Vector{Float64}` of extra points placed before and after the points given in
`times`. Useful for adding extra savepoints around discontinuos changes in an ODE.
Inludes the time points given in `times` themselves.
"""
function extra_save_points(times::Vector{Float64}; n = 5, window = 5.0)

    ret = Vector{Float64}()

    density = n / window

    for t in times

        t1 = max(0, t - window/2)
        t2 = t + window/2

        pts_before = collect(t1:density:t)
        pts_after  = collect(t:density:t2)
        append!(ret, union(pts_before, pts_after))
    end

    return ret
end

function sims(
    traits, 
    fwm, 
    prob;
    extinction_order = shuffle(species(fwm)),
    extinction_times = missing
    )

    s = richness(fwm) 
    n_extinctions = length(extinction_times)
    ntrajectories = 5
    g1 = 0.0
    g2 = 1.0
    stepsize = (g2 - g1)/ntrajectories
    gs = collect(g1:stepsize:g2)

    primary_extinctions = [Vector{Tuple{Float64, Symbol}}() for i in 1:ntrajectories]
    secondary_extinctions = [Vector{Tuple{Float64, Symbol}}() for i in 1:ntrajectories]

    function prob_func(prob, i, repeat)

        fwm = prob.f.sys

        es = ExtinctionSequenceCallback(fwm, deepcopy(extinction_order), extinction_times;
            extinction_history = primary_extinctions[i]
        );
        et = ExtinctionThresholdCallback(fwm, 10e-20;
            extinction_history = secondary_extinctions[i] 
        );
        cb = CallbackSet(et, es);

        return remake(prob, 
            callback = cb,
            p = Dict(:g => gs[i]) 
        )
    end

    function out_func(sol, i)

        prefs = [median_interaction_strength(sol, t) for t in sol.t]
        pop_cvs = time_window_population_cv(sol, 10.0;
            window_alignment = :LEFT
        )
        com_cvs = time_window_community_cv(sol, 5.0;
            window_alignment = :LEFT
        )
        

        ret = (
            sol = sol,
            t = sol.t, 
            primary_extinctions = primary_extinctions[i],
            secondary_extinctions = secondary_extinctions[i],
            richness = richness(sol),
            median_alpha = prefs, 
            pop_cv = pop_cvs, 
            com_cv  = com_cvs,
            g = gs[i]
        )

        return (ret, false)
    end

    eprob = EnsembleProblem(prob;
        prob_func = prob_func,
        output_func = out_func
    )

    @time sols = solve(eprob, 
        AutoTsit5(Rosenbrock23()), 
        EnsembleThreads(); 
        force_dtmin = true,
        maxiters = 1e7,
        tstops = extra_save_points(extinction_times), 
        trajectories = ntrajectories, 
        tspan = (1, 10_000 * n_extinctions + 1_000)
    );

    return sols
end

function index_of_time(sol, t)

    return findfirst(x -> x == t, sol.t)
end

function extinction_indices(sol)

    indices = Vector{Tuple{Symbol, Int64, Int64}}()

    for (i, (t, sp)) in enumerate(sol.primary_extinctions)
      
        i1 = index_of_time(sol, t)        

        if i + 1 <= length(sol.primary_extinctions)

            t_next, sp_next = sol.primary_extinctions[i + 1]            
            i2 = index_of_time(sol, t_next)
        else

            i2 = lastindex(sol.t)
        end

        push!(indices, (sp, i1, i2))
    end

    return indices
end

function cutup!(df, sols, foodweb_id = missing, sequence_id = missing)

    for sol in sols

        idxs = extinction_indices(sol)

        for (sp, i1, i2) in idxs 

            push!(df, (
                retcode = sol.sol.retcode,
                foodweb_id = foodweb_id,
                sequence_id = sequence_id,
                g = sol.g,
                extinction_species = sp,
                richness_pre = sol.richness[i1-1],
                secondary_extinctions = secondary_extinctions_over_period(sol, i1, i2),
                t1 = sol.t[i1],
                t2 = sol.t[i2],
                median_alpha = sol.median_alpha[i1:i2],
                pop_cv = sol.pop_cv[i1:i2],
                com_cv = sol.com_cv[i1:i2]
            ))

        end

    end

    return df
end

function secondary_extinctions_over_period(sol, id1, id2)

    secondary_extinctions = 0
    t1 = sol.t[id1]
    t2 = sol.t[id2]

    for (t, sp) in sol.secondary_extinctions

        if t1 < t < t2

            secondary_extinctions += 1
        end
    end

    return secondary_extinctions
end

stuff = DataFrame()

for fwm_num in 1:2

    traits, fwm, prob = build_my_fwm(20, 0.2, 2, 1.0);

    for seq_num in 1:10

        seq = (shuffle ∘ species)(fwm)
        times = collect((1_000.0:1_000.0:(length(seq) * 1_000.0)))

        sols = sims(traits, fwm, prob;
            extinction_times = times,
            extinction_order = seq
        );

        stuff = cutup!(stuff, sols, fwm_num, seq_num)
    end
end
