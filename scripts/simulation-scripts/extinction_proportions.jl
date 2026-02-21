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

@info "Dependencies Loaded"

# probably 4-5 in practice
const LARGE_CASCADE_CUTOFF = 2
const SAVE_PROBABILITY = 0.01

function process_solution(
    sol,   
    g, 
    primary_extinctions, 
    secondary_extinctions,
    foodweb_number,
    sequence_number,
    output_dir
)

    idxs = extinction_indices(sol, primary_extinctions)
    df = DataFrame()
    allowmissing!(df)

    richness_sol = richness(sol)
    fwm = sol.prob.f.sys

    # Run for each cascade in the run.
    for (sp, i1, i2) in idxs

        t1 = sol.t[i1]
        t2 = sol.t[i2]

        net = realized_network(sol, t1)
        trial = secondary_extinctions_during_trial(secondary_extinctions, t1, t2)

        # Save the state of the sims to disk if this is a large cascade, so we
        # can study it in more depth later.
        # All of the large ones. And a small random chance of saving not-large
        # ones so that we have smaller cascades for comparison.
        if ((richness_sol[i1-1] - richness_sol[i2-1]) > LARGE_CASCADE_CUTOFF) | (rand() < SAVE_PROBABILITY)

            mkpath(output_dir * "/realized_webs")
            jldsave(
                    output_dir *
                    "/realized_webs/foodweb_fw$(foodweb_number)_seq$(sequence_number)_sp$(sp).jld2",
                    true; 
                    fwm = fwm,
                    u_pre = sol[i1-1],
                    extinction_target = sp
            )
        end

        push!(df, (
            foodweb_number = foodweb_number,
            sequence_number = sequence_number,
            retcode = sol.retcode,
            g = g,
            extinction_target = sp, 
            vulnerability_target = vulnerability(net, sp),
            generality_target = generality(net, sp),
            richness_pre = richness_sol[i1-1],
            richness_post = richness_sol[i2-1],
            secondary_extinctions = length(trial),
            cascade_timespan = cascade_timespan(secondary_extinctions, t1, t2),
            avg_extinction_time = mean_extinction_time(secondary_extinctions, t1, t2),
            cascade_trophic_range = cascade_trophic_range(net, last.(trial)),
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
    ntrajectories = 10,
    g1 = 0.0,
    g2 = 0.5,
    time_between_extinctions = 1_000.0,
    stem = "niche-model"
    )

    output_dir = make_output_dir!(stem)

    save_parameters!(output_dir;
        species_richness = species_richness,
        connectance = connectance,
        minimum_basal_species = minimum_basal_species,
        number_of_foodwebs = number_of_foodwebs,
        number_of_sequences = number_of_sequences,
        ntrajectories = ntrajectories,
        g1 = g1,
        g2 = g2,
        time_between_extinctions = time_between_extinctions
    )

    for fwm_num in 1:number_of_foodwebs

        @info "Assembeling FoodwebModel $fwm_num of $number_of_foodwebs"

        web = niche_model_min_basal(species_richness, connectance, minimum_basal_species)

        traits, fwm = build_fwm(web, 0.2)
        prob = ODEProblem(fwm)
        prob = assemble_foodweb(prob;
            solver = Tsit5(),
            extra_transient_time = 1_000
        )

        mkpath(output_dir * "/initial_webs")
        jldsave(output_dir * "/initial_webs/foodweb_$fwm_num.jld2", true; 
            web = fwm.hg, traits = traits
        )

        @info "Assembled FoodwebModel $fwm_num"

        for seq_num in 1:number_of_sequences

            @info "Running FoodwebModel $fwm_num, sequence number $seq_num of $number_of_sequences"

            seq = (shuffle ∘ species)(fwm)
            times = collect((1_000.0:time_between_extinctions:(length(seq) * 1_000.0)))

            ps(x, y, z, w) = process_solution(x, y, z, w, fwm_num, seq_num, output_dir)

            sols = simulation_batch(fwm, prob, ps;
                extinction_times = times,
                extinction_order = seq,
                g1 = g1,
                g2 = g2,
                ntrajectories = ntrajectories
            );

            data = vcat(sols...)

            if (fwm_num == 1) & (seq_num == 1)

                CSV.write(output_dir * "/data.csv", data)
            else

                CSV.write(output_dir * "/data.csv", data; append = true)
            end
        end
    end

    @info "Done!"
end

simulations(
    species_richness = 10,
    minimum_basal_species = 1,
    number_of_foodwebs = 2,
    number_of_sequences = 2,
    stem = "test",
)