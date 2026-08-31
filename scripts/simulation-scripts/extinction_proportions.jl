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

    primary_extinctions = [
        Vector{Tuple{Float64, Vector{Symbol}}}() for i in 1:ntrajectories]
    secondary_extinctions = [
        Vector{Tuple{Float64, Vector{Symbol}}}() for i in 1:ntrajectories]

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

    @time sols = solve(eprob,
        Tsit5(),
        EnsembleThreads();
        force_dtmin = true,
        maxiters = 1e7,
        tstops = extinction_times,
        trajectories = ntrajectories,
        tspan = (1, maximum(extinction_times) + 1_000)
    );

    return sols
end

function process_solution(
    sol,   
    g, 
    primary_extinctions, 
    secondary_extinctions,
    foodweb_number,
    sequence_number,
    output_file,
    output_file_lock
)

    idxs = extinction_indices(sol, primary_extinctions)
    df = DataFrame()
    allowmissing!(df)

    richness_sol = richness(sol)

    # Run for each cascade in series.
    for (cascade_number, (sp, i1, i2)) in enumerate(idxs)

        t1 = sol.t[i1]
        t2 = sol.t[i2]

        net = realized_network(sol, t1)
        trial = secondary_extinctions_during_trial(secondary_extinctions, t1, t2)

        extinctions = collect(Iterators.flatten(last.(trial)))

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
            sequence_number = sequence_number,
            cascade_number = cascade_number,
            retcode = sol.retcode,
            g = g,
            target_species = sp, 
            cascade = trial,
            realized_web = realized_web_path,
            richness_pre = richness_sol[i1-1],
            richness_post = richness_sol[i2-1],
            cascade_trophic_range = cascade_trophic_range(net, extinctions),
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
    number_of_sequences = 10,
    n_extinctions = 1,
    max_trials = 5,
    ntrajectories = 10,
    g1 = 0.0,
    g2 = 0.5,
    time_between_extinctions = EXTINCTION_INTERVAL,
    stem = "niche-model"
    )

    output_file = make_output_file!(stem)
    output_file_lock = ReentrantLock()

    parameters = Dict(
        :species_richness => species_richness,
        :connectance => connectance,
        :minimum_basal_species => minimum_basal_species,
        :number_of_foodwebs => number_of_foodwebs,
        :number_of_sequences => number_of_sequences,
        :n_extinctions => n_extinctions,
        :ntrajectories => ntrajectories,
        :g1 => g1,
        :g2 => g2,
        :time_between_extinctions => time_between_extinctions
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

        for seq_num in 1:number_of_sequences

            @info "Running FoodwebModel $fwm_num, sequence number $seq_num of $number_of_sequences"

            seq = (shuffle ∘ species)(fwm)[1:max_trials]
            times = collect(
                (1_000.0:time_between_extinctions:(length(seq) * time_between_extinctions))
            )

            ps(x, y, z, w) = process_solution(x, y, z, w, 
                fwm_num, 
                seq_num, 
                output_file, 
                output_file_lock
            )

            sols = simulation_batch(fwm, prob, ps;
                extinction_times = times,
                extinction_order = seq,
                g1 = g1,
                g2 = g2,
                ntrajectories = ntrajectories,
                n_extinctions = n_extinctions
            );

            csvpath = output_file[1:end-4] * "csv"

            data = vcat(sols...)
          
            if (fwm_num == 1) & (seq_num == 1)

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
    minimum_basal_species = 1,
    number_of_foodwebs = 1,
    number_of_sequences = 1,
    max_trials = 5,
    n_extinctions = 1,
    stem = "test",
)