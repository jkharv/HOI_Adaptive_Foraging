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
    ntrajectories = 15
    traits, fwm, prob = build_my_fwm(s, 0.3, 2, 0.5);
    primary_extinctions = Vector{Tuple{Float64, Symbol}}()

    es = ExtinctionSequenceCallback(fwm, shuffle(species(fwm)), 5000.0);
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

        prefs = [interaction_strength(sol, t) for t in sol.t]

        ret = (sol.t, prefs, i/ntrajectories * 0.5555555555)
        return (ret, false)
    end

    eprob = EnsembleProblem(prob;
        prob_func = prob_func,
        output_func = out_func,
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


function interaction_strength(sol, t)

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

function time_covariance(sol, t)

    fwm = sol.prob.f.sys
    spp = variables(fwm, type = SPECIES_VARIABLE)



end


sol(1:sol.t[end], idxs = spp[1])


fwm, sols = sims();

f = WGLMakie.Figure()
ax = WGLMakie.Axis(f[1,1], xlabel = "time", ylabel = "median foraging preference")
empty!(ax)

for (t, prefs, g) in sols 
    
    WGLMakie.lines!(ax, t, prefs)
end