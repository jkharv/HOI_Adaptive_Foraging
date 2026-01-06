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
using Dates
using JLD2

include("my_model.jl")

OUTPUT_DIR = ""

@info "Dependencies Loaded"

SpeciesInteractionNetworks.CENTRALITY_MAXITER = 200

function simulation_batch(
    traits, 
    fwm, 
    prob;
    extinction_order = shuffle(species(fwm)),
    extinction_times = missing,
    g1 = 0.0,
    g2 = 0.5,
    ntrajectories = 5
    )
 
    n_extinctions = length(extinction_times)  
    stepsize = (g2 - g1)/(ntrajectories - 1)
    gs = collect(g1:stepsize:g2)

    primary_extinctions = [Vector{Tuple{Float64, Symbol}}() for i in 1:ntrajectories]
    secondary_extinctions = [Vector{Tuple{Float64, Symbol}}() for i in 1:ntrajectories]

    function prob_func(prob, i, repeat)

        fwm = prob.f.sys

        es = ExtinctionSequenceCallback(fwm, deepcopy(extinction_order), extinction_times;
            extinction_history = primary_extinctions[i]
        );
        et = ExtinctionThresholdCallback(fwm, 1e-15;
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
        tspan = (1, 10_000 * n_extinctions + 1_000)
    );

    return sols
end

function index_of_time(sol, t)

    return findfirst(x -> x == t, sol.t)
end

function count_secondary_extinctions(secondary_extinctions, t1, t2)

    return count(t -> (t[1] > t1) & (t[1] < t2), secondary_extinctions)
end

function extinction_indices(sol, primary_extinctions)

    indices = Vector{Tuple{Symbol, Int64, Int64}}()

    for (i, (t, sp)) in enumerate(primary_extinctions)
      
        i1 = index_of_time(sol, t)        

        if i + 1 <= length(primary_extinctions)

            t_next, sp_next = primary_extinctions[i + 1]            
            i2 = index_of_time(sol, t_next)
        else

            i2 = lastindex(sol.t)
        end

        push!(indices, (sp, i1, i2))
    end

    return indices
end

function centrality_of_spp(sol, t, sp)

    net = realized_network(sol, t)
    cd = centrality(EigenvectorCentrality, net)

    return cd[sp]
end

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
            centrality_primary = centrality(EigenvectorCentrality, net)[sp],
            vulnerability_primary = vulnerability(net, sp),
            generality_primary = generality(net, sp),
            richness_pre = richness_sol[i1-1],
            richness_post = richness_sol[i2-1],
            secondary_extinctions = count_secondary_extinctions(secondary_extinctions, t1, t2),
            t1 = sol.t[i1],
            t2 = sol.t[i2],
        ))
    end

    return df
end

function make_output_dir!()

    global OUTPUT_DIR = "sim-output/niche-model-" * string(Dates.today())
    mkdir(OUTPUT_DIR)

    return
end

function save_parameters!(; kwargs...)

    open(OUTPUT_DIR * "/simulation-parameters.txt", "a") do io 
        for args in kwargs

            println(io, args)
        end
    end

    return
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

    make_output_dir!()

    save_parameters!(
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

        traits, fwm = build_my_fwm(web, 0.2)
        prob = ODEProblem(fwm)
        prob = assemble_foodweb(prob;
            solver = Tsit5(),
            extra_transient_time = 1_000
        )

        jldsave(OUTPUT_DIR * "/foodweb_$fwm_num.jld2", true; 
            web = fwm.hg, traits = traits
        )

        @info "Assembled FoodwebModel $fwm_num"

        for seq_num in 1:number_of_sequences

            @info "Running FoodwebModel $fwm_num, sequence number $seq_num of $number_of_sequences"

            seq = (shuffle ∘ species)(fwm)
            times = collect((1_000.0:time_between_extinctions:(length(seq) * 1_000.0)))

            sols = simulation_batch(traits, fwm, prob;
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
                
                CSV.write(OUTPUT_DIR * "/data.csv", data)
            else

                CSV.write(OUTPUT_DIR * "/data.csv", data; append = true)
            end
        end
    end
    
    @info "Done!"
end

simulations(
    species_richness = 20,
    minimum_basal_species = 3,
    number_of_foodwebs = 2,
    number_of_sequences = 5
)

import WGLMakie

df = CSV.read("sim-output/niche-model-2025-12-05/data.csv", DataFrame)

# Extremely low richness is v noisy.
filter!(:richness_pre => x-> x >= 10, df)
# Some (very few) of the simulation end early because of instability or
# something. We can just exclude those to be safe. Including them or not didn't
# change any results. 
filter!(:retcode => x -> x == "Success", df)
# Add a column for proportion of community gone extinct after a primary extinction.
f(x, y) =  y ./ x
transform!(df, [:richness_pre, :secondary_extinctions] => f => :extinction_proportion)

fig = WGLMakie.Figure()
ax  = WGLMakie.Axis(fig[1,1])
WGLMakie.scatter!(ax, 
    df[:, :vulnerability_primary], 
    df[:, :extinction_proportion],
    color = df[:, :g]
)

names(df)