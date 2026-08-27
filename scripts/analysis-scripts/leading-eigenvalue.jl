include("../../src/HOI_Adaptive_Foraging.jl")

using .HOI_Adaptive_Foraging

using WGLMakie
using DataFrames
using CSV
using JLD2
using GLM
using LsqFit
using StatsBase
using Distributions
using Statistics
using SpeciesInteractionNetworks 
using DataStructures
using HigherOrderFoodwebs

# df1 = csv.read("sim-output/long-1-extinction-2026-07-30.csv", dataframe)
# jld1 = jldopen("sim-output/long-1-extinction-2026-07-30.jld2")
df4 = CSV.read("sim-output/long-4-extinctions-2026-07-30.csv", DataFrame)
jld4 = jldopen("sim-output/long-4-extinctions-2026-07-30.jld2")

preprocessing!(df4, jld4)

# df = vcat(df1, df4)
df = df4

filt = copy(df)
filter!(:cascade_number => x->x==1, filt)

fig = Figure(size = (800, 800))
ax  = Axis(fig[1, 1], xlabel = "Cascade Size", ylabel = "Leading Eigenvalue")
scatter!(ax, 
    filt[:, :extinction_proportion], 
    filt[:, :real_eigenvalue]
)

res = lm(@formula(real_eigenvalue ~ extinction_proportion), filt)

abline!(ax, coef(res)[1], coef(res)[2]; color = :red, linewidth = 6)

#
# λ v g
# No Effect

fig = Figure(size = (800, 800))
ax  = Axis(fig[1, 1], xlabel = "Adaptation Rate", ylabel = "Leading Eigenvalue")
scatter!(ax, 
    filt[:, :g], 
    filt[:, :real_eigenvalue]
)

res = lm(@formula(real_eigenvalue ~ g), filt)