using SpeciesInteractionNetworks
using HigherOrderFoodwebs
using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

include("../src/HOI_Adaptive_Foraging.jl")
include("../scripts/my_model.jl")

using .HOI_Adaptive_Foraging
using OrdinaryDiffEqTsit5

using LinearAlgebra
using Statistics
using DataFrames
using StatsBase

import WGLMakie

function process_solution(sol, g)

    fwm = sol.prob.f.sys

    trait_var_t = timescale.(Ref(sol), variables(fwm, type = TRAIT_VARIABLE))
    density_var_t = timescale.(Ref(sol), variables(fwm, type = SPECIES_VARIABLE)) 

    cat = vcat([:TRAIT for i in 1:length(trait_var_t)], [:DENSITY for i in 1:length(density_var_t)])
    vars = vcat(trait_var_t, density_var_t)
    g_val = [g for i in 1:length(vars)] 

    return DataFrame(category = cat, dynamic_timescale = vars, gval = g_val)
end

function process_solution_mean(sol, g)

    fwm = sol.prob.f.sys

    trait_var_t = mean(timescale.(Ref(sol), variables(fwm, type = TRAIT_VARIABLE)))
    density_var_t = mean(timescale.(Ref(sol), variables(fwm, type = SPECIES_VARIABLE)))

    cat = [:TRAIT, :DENSITY]
    vars = [trait_var_t, density_var_t]
    g_val = [g for i in 1:length(vars)] 

    return DataFrame(category = cat, dynamic_timescale = vars, gval = g_val)
end

function timescale(sol, idx)

    index = get_index(sol.prob.f.sys.vars, idx) 

    x  = abs.(sol(sol.t[(Integer ∘ floor)(end/2):end], Val{0}; idxs = index))
    dx = abs.(sol(sol.t[(Integer ∘ floor)(end/2):end], Val{1}; idxs = index))

    t  = dx ./ x

    for (i,e) in enumerate(t)

        if isnan(e)

            t[i] = 0.0
        end
    end

    return mean(t)
end

df = DataFrame()

for i in 1:10

    traits, fwm = build_my_fwm(10, 0.3, 5, 0.2)

    prob = ODEProblem(fwm)
    prob = assemble_foodweb(prob; solver = Tsit5(), extra_transient_time = 1_000)

    g1 = 0.0
    g2 = 1.0
    ntrajectories = 20

    stepsize = (g2 - g1)/(ntrajectories - 1)
    gs = collect(g1:stepsize:g2)

    function prob_func(prob, i, repeat)

        fwm = prob.f.sys

        et = ExtinctionThresholdCallback(fwm, 1e-15);
        am = AlphaManifoldCallback(fwm);
        cb = CallbackSet(et, am);

        return remake(prob, 
            callback = cb,
            p = Dict(:g => gs[i]),
        )
    end

    function out_func(sol, i)

        out = (sol, gs[i])

        return (out, false)
    end

    eprob = EnsembleProblem(prob;
        prob_func = prob_func,
        output_func = out_func
    )

    @time sols = solve(eprob, 
        Tsit5(), 
        EnsembleThreads(); 
        force_dtmin = true,
        maxiters = 1e7,
        trajectories = ntrajectories, 
        tspan = (1, 2000)
    );

    for sol in sols

        d = process_solution_mean(sol...)
        append!(df, d)
    end
end

x = filter(:category => x -> x==:TRAIT, df)
x = df

# f  = WGLMakie.Figure(width = 1920, height = 1080)
# ax = WGLMakie.Axis(f[1,1], xlabel = "g", ylabel = "timmescale") 

empty!(ax)

WGLMakie.scatter!(ax, x[:, :gval], x[:, :dynamic_timescale], color = :black)

# The graph isn't super interpretable and it also doesn't show a saturation in
# the timescale of foraging dynamics. This could make sense because, maybe the
# derivative keeps getting large because of g but the resultant change in
# biomass uptake could be small. Moreover, I have to do the whole alpha manifold 
# thing to keep those numbers bounded, so it's not clear that numerically speaking
# I'm even measuring what I want by doing dy/dt / y

# I'll need to rethink how I'm measuring the "timescale" of the foraging dynamics.