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
using StatsBase
using CSV

include("my_model.jl")

using Profile

@profview fwm = build_my_fwm(5, 0.3, 1, 0.3);


