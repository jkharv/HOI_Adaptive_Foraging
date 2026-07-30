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

        # Just after the extinction occurs
        λ = leading_eigenvalue(sol, sol.t[i1 + 1])

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
            real_eigenvalue = real(λ),
            imag_eigenvalue = imag(λ),
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
            extra_transient_time = 1_000
        )

        jldopen(output_file, "a+", compress = ZstdFilter()) do f
            f["initial_webs/foodweb_$fwm_num"] = fwm.hg
            f["traits/traits_$fwm_num"] = traits
        end

        @info "Assembled FoodwebModel $fwm_num"

        for seq_num in 1:number_of_sequences

            @info "Running FoodwebModel $fwm_num, sequence number $seq_num of $number_of_sequences"

            seq = (shuffle ∘ species)(fwm)
            times = collect((1_000.0:time_between_extinctions:(length(seq) * 1_000.0)))

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
    n_extinctions = 1,
    stem = "test",
)