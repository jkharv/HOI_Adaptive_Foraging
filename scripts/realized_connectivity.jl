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
using CairoMakie
using CSV

include("my_model.jl")


function realized_connectivity(c, g)

    fwm = build_my_fwm(10, 0.3, 2, 0.4);
    prob = ODEProblem(fwm);
    
    sol = solve(prob, AutoTsit5(Rosenbrock23());
        tspan = (1, 2000)
    );

    intxs = trophic_flux(fwm, sol, 2000; include_loops = false)
    c = count(!iszero, values(intxs)) / 10^2
    return c
end

function mean_interaction_strength(c, g)

    fwm = build_my_fwm(10, 0.3, 2, 0.4);
    prob = ODEProblem(fwm);
    
    sol = solve(prob, AutoTsit5(Rosenbrock23());
        tspan = (1, 2000)
    );

    intxs = trophic_flux(fwm, sol, 1500, 2000; include_loops = false)

    return (abs ∘ mean ∘ values)(intxs)
end

df = DataFrame(
    :g => Vector{Float64}(), 
    :nominal_c => Vector{Float64}(),
    :mean_interaction_strength => Vector{Float64}()
)

for gval in 0.0:0.01:0.25

    s = mean_interaction_strength(0.3, gval)

    push!(df,(
        g = gval,
        nominal_c = 0.3,
        mean_interaction_strength = s
    ))
end

# Not any apparrent difference in realized connectivity accross different g values. :(
# Actually now there is. It seems that the transient time for attack rates is longer
# than for biomass, and I just had to simulate for longer. Fiddling with other stuff
# now too.
scatter(df.g, df.mean_interaction_strength)