include("../src/HOI_Adaptive_Foraging.jl")

using .HOI_Adaptive_Foraging

using OrdinaryDiffEqTsit5

using SpeciesInteractionNetworks
using HigherOrderFoodwebs
using AnnotatedHypergraphs
using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

using Random
using LinearAlgebra
using Statistics
using Distributions
using DataFrames
using StatsBase
using CSV
using JLD2

include("my_model.jl")

@info "Dependencies Loaded"

function process_solution(sol, g, primary_extinctions, secondary_extinctions)

    idxs = extinction_indices(sol, primary_extinctions)
    df = DataFrame()

    richness_sol = richness(sol)
    fwm = sol.prob.f.sys
    hg = fwm.hg

    for (sp, i1, i2) in idxs

        t1 = sol.t[i1]
        t2 = sol.t[i2]

        net = realized_network(sol, t1)

        push!(df, (
            retcode = sol.retcode,
            g = g,
            extinction_species = sp, 
            vulnerability_primary = vulnerability(net, sp),
            generality_primary = generality(net, sp),
            richness_pre = richness_sol[i1-1],
            richness_post = richness_sol[i2-1],
            secondary_extinctions = count_secondary_extinctions(secondary_extinctions, t1, t2),
            timespan_of_cascade = cascade_timespan(secondary_extinctions, t1, t2),
            t1 = sol.t[i1],
            t2 = sol.t[i2]
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
    time_between_extinctions = 1_000.0
    )

    output_dir = make_output_dir!("niche-model")

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

        jldsave(output_dir * "/foodweb_$fwm_num.jld2", true; 
            web = fwm.hg, traits = traits
        )

        @info "Assembled FoodwebModel $fwm_num"

        for seq_num in 1:number_of_sequences

            @info "Running FoodwebModel $fwm_num, sequence number $seq_num of $number_of_sequences"

            seq = (shuffle ∘ species)(fwm)
            times = collect((1_000.0:time_between_extinctions:(length(seq) * 1_000.0)))

            sols = simulation_batch(fwm, prob, process_solution;
                extinction_times = times,
                extinction_order = seq,
                g1 = g1,
                g2 = g2,
                ntrajectories = ntrajectories
            );

            data = vcat(sols...)
            data[!, :foodweb_number] .= fwm_num
            data[!, :sequence_number] .= seq_num 

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
    number_of_sequences = 2
)