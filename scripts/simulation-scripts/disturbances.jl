include("../../src/HOI_Adaptive_Foraging.jl")

using .HOI_Adaptive_Foraging

using OrdinaryDiffEqTsit5

using SpeciesInteractionNetworks
using HigherOrderFoodwebs
using AnnotatedHypergraphs

using Random
using LinearAlgebra
using Statistics
using Distributions
using DataFrames
using StatsBase
using CSV
using JLD2
using UUIDs

@info "Dependencies Loaded"

function simulation_batch(fwm, prob, process_solution::Function;
    disturbance_times = missing,
    n_disturbances = 1,
    removal_fraction = 0.5,
    g1 = 0.0,
    g2 = 0.5,
    ntrajectories = 5
    )

    @assert ntrajectories > 2
    @assert !ismissing(disturbance_times)
    @assert g1 < g2

    stepsize = (g2 - g1) / (ntrajectories - 1)
    gs = collect(g1:stepsize:g2)

    disturbances = [ Vector{Tuple{Float64, Vector{Symbol}}}() for i in 1:ntrajectories ]
    extinctions  = [ Vector{Tuple{Float64, Vector{Symbol}}}() for i in 1:ntrajectories ]

    # This function is responsible for taking and index into the batch, `i`, and
    # returning the `ODEProblem` that is to be run. `repeat` is unused but is
    # required by the `EnsembleProblem` API.
    function prob_func(prob, i, repeat)

        fwm = prob.f.sys
        
        dc = DisturbanceCallback(fwm, n_disturbances, disturbance_times;
            removal_fraction = removal_fraction,
            disturbance_history = disturbances[i]
        );
        et = ExtinctionThresholdCallback(fwm, EXTINCTION_THRESHOLD;
            extinction_history = extinctions[i]
        );
        am = AlphaManifoldCallback(fwm);
        cb = CallbackSet(et, dc, am);

        return remake(prob,
            callback = cb,
            p = Dict(:g => gs[i]),
        )
    end

    function out_func(sol, i)

        ret = process_solution(
            sol,
            gs[i],
            disturbances[i],
            extinctions[i]
        )

        return (ret, false)
    end

    eprob = EnsembleProblem(prob;
        prob_func = prob_func,
        output_func = out_func
    )

    @time sols = solve(eprob,
        Tsit5(),
        EnsembleSerial();
        force_dtmin = true,
        maxiters = 1e7,
        tstops = disturbance_times,
        trajectories = ntrajectories,
        tspan = (1, maximum(disturbance_times) + 1_000)
    );

    return sols
end

function process_solution(
    sol,   
    g, 
    disturbances, 
    extinctions,
    foodweb_number,
    run_number,
    output_file,
    output_file_lock
)

    idxs = extinction_indices(sol, disturbances)
    df = DataFrame()
    allowmissing!(df)

    richness_sol = richness(sol)

    # Run for each cascade in series.
    for (cascade_number, (sp, i1, i2)) in enumerate(idxs)

        t1 = sol.t[i1]
        t2 = sol.t[i2]

        net = realized_network(sol, t1)
        trial = secondary_extinctions_during_trial(extinctions, t1, t2)

        extinctions_during_trial = collect(Iterators.flatten(last.(trial)))

        trial_id = (string ∘ uuid4)() 
        realized_web_path = "realized_webs/$trial_id"
        lock(output_file_lock) do
            jldopen(output_file, "a+", compress = ZstdFilter()) do f 
                f[realized_web_path] = net
                f["densities/$trial_id"] = sol[i1-1]
            end
        end

        push!(df, (
            foodweb_number = foodweb_number,
            run_number = run_number,
            cascade_number = cascade_number,
            retcode = sol.retcode,
            g = g,
            target_species = sp, 
            cascade = trial,
            realized_web = realized_web_path,
            richness_pre = richness_sol[i1-1],
            richness_post = richness_sol[i2-1],
            cascade_trophic_range = cascade_trophic_range(net, extinctions_during_trial),
            maximum_trophic_level = maximum_trophic_level(net),
            t1 = t1,
            t2 = t2 
        )
        )
    end

    return df
end

function simulations(;
    species_richness = 20,
    connectance = 0.3,
    minimum_basal_species = 5,
    number_of_foodwebs  = 5,
    removal_fraction = 0.5,
    number_of_runs = 10,
    n_disturbances = 1,
    ntrajectories = 10,
    g1 = 0.0,
    g2 = 0.5,
    time_between_disturbances = EXTINCTION_INTERVAL,
    stem = "disturbances"
    )

    output_file = make_output_file!(stem)
    output_file_lock = ReentrantLock()

    parameters = Dict(
        :species_richness => species_richness,
        :connectance => connectance,
        :minimum_basal_species => minimum_basal_species,
        :removal_fraction => removal_fraction,
        :number_of_foodwebs => number_of_foodwebs,
        :number_of_runs => number_of_runs,
        :n_disturbances => n_disturbances,
        :ntrajectories => ntrajectories,
        :g1 => g1,
        :g2 => g2,
        :time_between_disturbances => time_between_disturbances
    )

    for fwm_num in 1:number_of_foodwebs

        @info "Assembeling FoodwebModel $fwm_num of $number_of_foodwebs"

        web = niche_model_min_basal(species_richness, connectance, minimum_basal_species)

        traits, fwm = build_fwm(web, 0.2)
        prob = ODEProblem(fwm)
        prob = assemble_foodweb(prob;
            solver = Tsit5(),
            extra_transient_time = 5_000
        )

        jldopen(output_file, "a+", compress = ZstdFilter()) do f
            f["initial_webs/foodweb_$fwm_num"] = fwm.hg
            f["traits/traits_$fwm_num"] = traits
        end

        @info "Assembled FoodwebModel $fwm_num"

        for run_num in 1:number_of_runs

            @info "Running FoodwebModel $fwm_num, run number $run_num of $number_of_runs"

            seq = (shuffle ∘ species)(fwm)
            times = collect((1_000.0:time_between_disturbances:(length(seq) * 1_000.0)))

            ps(x, y, z, w) = process_solution(x, y, z, w, 
                fwm_num, 
                run_num,
                output_file, 
                output_file_lock
            )

            sols = simulation_batch(fwm, prob, ps;
                disturbance_times = times,
                n_disturbances = n_disturbances,
                removal_fraction = removal_fraction,
                g1 = g1,
                g2 = g2,
                ntrajectories = ntrajectories,
            );

            csvpath = output_file[1:end-4] * "csv"

            data = vcat(sols...)
          
            if (fwm_num == 1) & (run_num == 1)

                CSV.write(csvpath, data)
            else

                CSV.write(csvpath, data; append = true)
            end
        end
    end

    @info "Done!"
end

simulations(
    species_richness = 30,
    minimum_basal_species = 4,
    number_of_foodwebs = 1,
    removal_fraction = 0.9,
    number_of_runs = 1,
    n_disturbances = 1,
    stem = "disturbances",
)