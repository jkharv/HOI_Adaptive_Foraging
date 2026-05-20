include("src/HOI_Adaptive_Foraging.jl")

using .HOI_Adaptive_Foraging

using OrdinaryDiffEqTsit5
using SpeciesInteractionNetworks
using HigherOrderFoodwebs
using DataFrames
using Random
using AnnotatedHypergraphs

@time for i in 1:10

    web = niche_model_min_basal(30, 0.3, 4);
    traits, fwm = build_fwm(web, 0.15);
    prob = ODEProblem(fwm)
    assemble_foodweb(prob, solver = Tsit5(), extra_transient_time = 1_000)

    seq = (shuffle ∘ species)(fwm)
    times = collect((1_000.0:8_000:(length(seq) * 1_000.0)))

    ps(x, y, z, w) = (x,y,z,w) -> DataFrame()

    sols = simulation_batch2(fwm, prob, ps;
        extinction_times = times,
        extinction_order = seq,
        g1 = 0.0,
        g2 = 0.5,
        ntrajectories = 10,
        n_extinctions = 1,
    );
end

function simulation_batch2(fwm, prob, process_solution::Function;
    extinction_order = shuffle(species(fwm)),
    extinction_times = missing,
    n_extinctions = 1,
    g1 = 0.0,
    g2 = 0.5,
    ntrajectories = 5
    )

    @assert ntrajectories > 2
    @assert !ismissing(extinction_times)
    @assert g1 < g2
    @assert extinction_order ⊆ species(fwm) 

    stepsize = (g2 - g1)/(ntrajectories - 1)
    gs = collect(g1:stepsize:g2)

    primary_extinctions = [Vector{Tuple{Float64, Vector{Symbol}}}() for i in 1:ntrajectories]
    secondary_extinctions = [Vector{Tuple{Float64, Vector{Symbol}}}() for i in 1:ntrajectories]

    # This function is responsible for taking and index into the batch, `i`, and
    # returning the `ODEProblem` that is to be run. `repeat` is unused but is
    # required by the `EnsembleProblem` API.
    function prob_func(prob, i, repeat)

        fwm = prob.f.sys

        es = ExtinctionSequenceCallback(fwm, 
            extinction_order, extinction_times;
            n_extinctions = n_extinctions,
            extinction_history = primary_extinctions[i]
        );
        et = ExtinctionThresholdCallback(fwm, EXTINCTION_THRESHOLD;
            extinction_history = secondary_extinctions[i]
        );
        am = AlphaManifoldCallback(fwm);
        cb = CallbackSet(et, es, am);

        return remake(prob,
            callback = cb,
            p = Dict(:g => gs[i]),
        )
    end

    function out_func(sol, i)

        ret = process_solution(
            sol,
            gs[i],
            primary_extinctions[i],
            secondary_extinctions[i]
        )

        return (ret, false)
    end

    eprob = EnsembleProblem(prob;
        prob_func = prob_func,
        output_func = out_func
    )

    tend = 8_000 + 100

    @time sols = solve(eprob,
        Tsit5(),
        EnsembleThreads();
        force_dtmin = true,
        maxiters = 1e7,
        tstops = extinction_times,
        trajectories = ntrajectories,
        tspan = (1, tend)
    );

    return sols
end