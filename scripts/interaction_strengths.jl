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

    s = 10 
    n_extinctions = s/2
    ntrajectories = 15
    traits, fwm, prob = build_my_fwm(s, 0.3, 2, 0.0);
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

        ret = (sol, i/ntrajectories * 0.5555555555)
        return (ret, false)
    end

    eprob = EnsembleProblem(prob;
        prob_func = prob_func,
        output_func = out_func,
        safetycopy = true
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

function normed_alphas(fwm, sol, t)

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

    return a_norms
end

fwm, sols = sims();

f = WGLMakie.Figure()
ax = WGLMakie.Axis(f[1,1], xlabel = "time", ylabel = "median foraging preference")

for (sol, g) in sols 

    abars = []
    ts = []
    for t in 0:100:5000 

        abar = median(normed_alphas(fwm, sol, t))

        push!(abars, abar)
        push!(ts, t)
    end

    WGLMakie.lines!(ax, ts, abars)
end

f = WGLMakie.Figure()
ax = WGLMakie.Axis(f[1,1], xlabel = "Timestep", ylabel = "Biomass")

sol = first(sols[5]);
for sp in species(fwm)
        
        WGLMakie.lines!(ax, sol.t, sol[sp])
end
