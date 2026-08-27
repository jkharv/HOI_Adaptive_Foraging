include("../../src/HOI_Adaptive_Foraging.jl")

using .HOI_Adaptive_Foraging

using WGLMakie
using DataFrames
using CSV
using JLD2
using GLM
using CategoricalArrays
using LsqFit
using StatsBase
using Distributions
using Statistics
using SpeciesInteractionNetworks 
using DataStructures
using HigherOrderFoodwebs

df1  = CSV.read("sim-output/disturbances-2026-08-27.csv", DataFrame)
jld1 = jldopen("sim-output/disturbances-2026-08-27.jld2")

preprocessing!(df1, jld1)

df = copy(df1)




