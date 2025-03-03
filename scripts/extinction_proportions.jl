include("../src/HOI_Adaptive_Foraging.jl")

using .HOI_Adaptive_Foraging
using OrdinaryDiffEq
using HigherOrderFoodwebs
using SpeciesInteractionNetworks
using ModelingToolkit
using Random
using LinearAlgebra
using Statistics
using Distributions
using DataFrames
using CSVFiles

include("my_model.jl")

function simulation_batch(s, c, b, gs)

    df = DataFrame(
        :primary_extinctions => Vector{Int64}(), 
        :secondary_extinctions => Vector{Int64}(),
        :g => Vector{Float64}() 
    );
    probs = []

    io_lock = ReentrantLock()

    for g ∈ gs

        fwm = build_my_fwm(s, c, b, g);
        prob = ODEProblem(fwm)

        push!(probs, (fwm, prob, g))
    end

    Threads.@threads for (fwm, prob, g) ∈ probs

        # Set up vectors to record extinction data in
        primary_extinctions = Vector{Tuple{Float64,Symbol}}()
        secondary_extinctions = Vector{Tuple{Float64,Symbol}}()

        # Set up the callbacks
        et = ExtinctionThresholdCallback(fwm, 1e-20; 
            extinction_history = secondary_extinctions);
        es = ExtinctionSequenceCallback(fwm, shuffle(species(fwm)), 250.0; 
            extinction_history = primary_extinctions);
        rt = RichnessTerminationCallback(fwm, 0.5);

        # Simulate
        solve(prob, AutoTsit5(Rosenbrock23());
            callback = CallbackSet(et, es, rt), 
            force_dtmin = true,
            save_on = false,
            maxiters = 1e6,
            tspan = (0, 5000)
        );

        lock(io_lock)

        push!(df, 
            (
            primary_extinctions = length(primary_extinctions),
            secondary_extinctions = length(secondary_extinctions),
            g = g
            )
        )

        unlock(io_lock)
    end

    return df
end

function simulations(s, c, b, gmin, gmax, n, batch_size, outpath)
 
    df = DataFrame(
        :primary_extinctions => Vector{Int64}(), 
        :secondary_extinctions => Vector{Int64}(),
        :g => Vector{Float64}() 
    );

    stepsize = (gmax - gmin) / (n-1);
    steps = gmin:stepsize:gmax
   
    batches  = Iterators.partition(steps, batch_size)

    for batch ∈ batches

        df_batch = simulation_batch(s, c, b, batch)
        append!(df, df_batch)


        println(df_batch)
    end

    return df
end

simulations(
    5, 0.3, 1, 0.0, 0.25, 100, 5, "./output/data.csv"
)