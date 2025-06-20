include("../src/HOI_Adaptive_Foraging.jl")

using .HOI_Adaptive_Foraging

using OrdinaryDiffEqTsit5
using OrdinaryDiffEqRosenbrock
using DiffEqNoiseProcess
using StochasticDiffEq

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

import WGLMakie
using FoodwebPlots
include("my_model.jl")
include("my_stochastic_model.jl")

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
    g2 = 0.5
    stepsize = (g2 - g1)/(ntrajectories - 1)
    gs = collect(g1:stepsize:g2)

    primary_extinctions = [Vector{Tuple{Float64, Symbol}}() for i in 1:ntrajectories]
    secondary_extinctions = [Vector{Tuple{Float64, Symbol}}() for i in 1:ntrajectories]

    function prob_func(prob, i, repeat)

        fwm = prob.f.sys

        es = ExtinctionSequenceCallback(fwm, deepcopy(extinction_order), extinction_times;
            extinction_history = primary_extinctions[i]
        );
        et = ExtinctionThresholdCallback(fwm, 1e-15;
            extinction_history = secondary_extinctions[i] 
        );
        am = AlphaManifoldCallback(fwm);
        cb = CallbackSet(et, es, am);

        noise = WienerProcess(0.0, zeros(Float64, size(prob.u0)), zeros(Float64, size(prob.u0)))

        return remake(prob, 
            callback = cb,
            p = Dict(:g => gs[i]),
            noise = noise
        )
    end

    function out_func(sol, i)

        prefs = median_interaction_strength(sol)
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
        SOSRA(), 
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
                richness_post = sol.richness[i2-1],
                secondary_extinctions = secondary_extinctions_over_period(sol, i1, i2),
                t1 = sol.t[i1],
                t2 = sol.t[i2],
                median_alpha = sol.median_alpha[i1:i2],
                pop_cv = sol.pop_cv[i1:i2],
                com_cv = sol.com_cv[i1:i2]
            ))
        end
    end
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

function do_stuff()

    stuff = DataFrame()

    for fwm_num in 1:1

        traits, fwm, prob = build_my_stochastic_fwm(10, 0.3, 2, 0.0);

        for seq_num in 1:3

            seq = (shuffle ∘ species)(fwm)
            times = collect((1_000.0:1_000.0:(length(seq) * 1_000.0)))

            sols = sims(traits, fwm, prob;
                extinction_times = times,
                extinction_order = seq
            );

            cutup!(stuff, sols, fwm_num, seq_num)
        end
    end

    CSV.write("data.csv", stuff)

end

do_stuff()

# df = CSV.read("data.csv", DataFrame)

# traits, fwm, prob = build_my_stochastic_fwm(20, 0.3, 1, 0.2);

# primary_extinctions = Vector{Tuple{Float64, Symbol}}()
# secondary_extinctions = Vector{Tuple{Float64, Symbol}}()

# times = collect((1_000.0:1_000.0:(richness(fwm) * 1_000.0)))
# es = ExtinctionSequenceCallback(fwm, (shuffle ∘ species)(fwm), times;
#     extinction_history = primary_extinctions
# );
# et = ExtinctionThresholdCallback(fwm, 1e-10;
#     extinction_history = secondary_extinctions
# );
# am = AlphaManifoldCallback(fwm);

# prob = remake(prob, 
#     p = Dict(:g => 0.15)
# )

# sol = @time sols = solve(prob, 
#     SOSRA(), 
#     callback = CallbackSet(am, et, es),
#     force_dtmin = true,
#     tstops = times,
#     maxiters = 1e7,
#     tspan = (1, 10_000)
# );

# f = WGLMakie.Figure()
# ax = WGLMakie.Axis(f[1,1], xlabel = "time", ylabel = "Biomass")
# empty!(ax)
# for sp in species(fwm)

#     WGLMakie.lines!(ax, sol.t, sol[sp])
# end

# species(fwm)

# WGLMakie.vlines!(ax, first.(primary_extinctions))
# WGLMakie.vlines!(ax, first.(secondary_extinctions))

# ax = WGLMakie.Axis(f[1,1], xlabel = "time", ylabel = "Preference")
# empty!(ax)
# for v in variables(fwm.vars, type = TRAIT_VARIABLE)

#     try

#         WGLMakie.lines!(ax, sol.t, sol[v])
#     catch

#         continue
#     end
# end

# foodwebplot(fwm.hg;
#     draw_loops = false,
#     trophic_levels = true,
#     node_weights = 5
# )

# richness(sol)

# eig = eigenstability(sol)


# WGLMakie.lines!(ax, sol.t, eig)