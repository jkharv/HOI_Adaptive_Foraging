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

import WGLMakie
include("my_model.jl")

function sims()

    s = 5
    n_extinctions = s/2
    ntrajectories = 5
    traits, fwm, prob = build_my_fwm(s, 0.3, 2, 0.5);
    primary_extinctions = Vector{Tuple{Float64, Symbol}}()

    # es = ExtinctionSequenceCallback(fwm, shuffle(species(fwm)), 5000.0);
    rt = RichnessTerminationCallback(fwm, 0.5);
    et = ExtinctionThresholdCallback(fwm, 10e-20);
    cb = CallbackSet(et, rt, es);

    function prob_func(prob, i, repeat)

        return remake(prob, 
            callback = deepcopy(cb),
            p = Dict(:g => (i-1)/ntrajectories * 0.555555555) 
        )
    end

    function out_func(sol, i)

        prefs = [median_interaction_strength(sol, t) for t in sol.t]
        pop_cvs = time_window_population_cv(sol, 10.0)
        com_cvs = time_window_community_cv(sol, 10.0)

        ret = (
            t = sol.t, 
            median_alpha = prefs, 
            pop_cv_avg = pop_cvs, 
            com_cv = com_cvs,
            g = i/ntrajectories * 0.5555555555
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
        trajectories = ntrajectories, 
        tspan = (1, n_extinctions * 5500.0)
    );

    return (fwm, sols)
end

# @time sol = solve(prob, 
#         AutoTsit5(Rosenbrock23()), 
#         callback = cb,
#         force_dtmin = true,
#         maxiters = 1e7,
#         trajectories = ntrajectories, 
#         tspan = (1, n_extinctions * 5500.0)
# );

fwm, sols = sims();

f = WGLMakie.Figure()
ax = WGLMakie.Axis(f[1,1], xlabel = "time", ylabel = "median foraging preference")
empty!(ax)

for sol in sols 
    
    WGLMakie.lines!(ax, sol.t, sol.com_cv)
end