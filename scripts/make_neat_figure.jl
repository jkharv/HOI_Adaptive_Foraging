include("../src/HOI_Adaptive_Foraging.jl")

using .HOI_Adaptive_Foraging
using DifferentialEquations
using HigherOrderFoodwebs
using SpeciesInteractionNetworks
using ModelingToolkit
using Random
using LinearAlgebra
using Statistics
using Distributions
using DataFrames
using CSVFiles
using SymbolicIndexingInterface
using FoodwebPlots
import WGLMakie

include("my_model.jl")

df = DataFrame(
    :primary_extinctions => Vector{Int64}(), 
    :secondary_extinctions => Vector{Int64}(),
    :g => Vector{Float64}() 
)

sol = 1
fwm = 1

for g in 0.0:0.0625:0.25

    fwm = build_my_fwm(20, 0.3, 5, g);
    prob = ODEProblem(fwm)

    # Set up vectors to record extinction data in
    primary_extinctions = Vector{Tuple{Float64,Symbol}}()
    secondary_extinctions = Vector{Tuple{Float64,Symbol}}()

    # Set up the callbacks
    et = HOI_Adaptive_Foraging.ExtinctionThresholdCallback2(fwm, 1e-20; 
        extinction_history = secondary_extinctions);
    es = ExtinctionSequenceCallback(fwm, shuffle(species(fwm)), 250.0; 
        extinction_history = primary_extinctions);
    rt = HOI_Adaptive_Foraging.RichnessTerminationCallback(fwm, 0.5);

    # Simulate
    sol = solve(prob, Rosenbrock23();
        callback = CallbackSet(et, es, rt), 
        force_dtmin = true,
        maxiters = 1e6,
        tspan = (0, 5000)
    );

    push!(df, 
        (
            primary_extinctions = length(primary_extinctions),
            secondary_extinctions = length(secondary_extinctions),
            g = g
        )
    )    
end