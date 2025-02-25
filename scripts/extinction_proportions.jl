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

function simulations(s, c, b, gmin, gmax, n, outpath)
 
    df = DataFrame(
        :primary_extinctions => Vector{Int64}(), 
        :secondary_extinctions => Vector{Int64}(),
        :g => Vector{Float64}() 
    );
    io_lock = ReentrantLock();

    snum = 1;
    stepsize = (gmax - gmin) / (n-1);
   
    probs = []

    for g ∈ gmin:stepsize:gmax

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

        # Why didn't they make data frames thread safe?
        lock(io_lock) do

            push!(df, 
                (
                    primary_extinctions = length(primary_extinctions),
                    secondary_extinctions = length(secondary_extinctions),
                    g = g
                )
            )    

            println("Finished simulation number $snum for g = $g")
            snum += 1

            save(outpath, df)
        end
    end
end

simulations(
    30, 0.3, 5, 0.0, 0.25, 100, "./output/data.csv"
)