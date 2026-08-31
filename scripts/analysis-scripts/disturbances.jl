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

transform!(df, [:alpha_web, :cascade] => ByRow(
    (x, y) -> begin

        spp = last.(y) 
        if isempty(spp)
            return 0.0
        end

        spp = reduce(vcat, spp)
        return mean(niche_breadth.(Ref(x), spp))
    end) 
    => :cascade_niche_breadth
)  

transform!(df, [:alpha_web, :density_pre] => ByRow(
    (x, y) -> begin

        # Find the non-zero species
        indxs = findall(!iszero, y)

        return mean(niche_breadth.(Ref(x), species(x)[indxs]))
    end) 
    => :community_niche_breadth
)

transform!(df, [:alpha_web] => 
    ByRow(niche_overlap) 
    => :community_niche_overlap
)

transform!(df, [:realized_prob] =>
    ByRow(connectance)
    => :connectance
)

#
#
#

filt = filter(:cascade_number => (x -> x < 5), df)

fig = Figure(size = (800, 800))
ax  = Axis(fig[1,1], xlabel = "Adaptation Rate", ylabel = "Niche Overlap")
boxplot!(ax, 
    filt[:, :g],
    filt[:, :community_niche_overlap],
    dodge = filt[:, :cascade_number],
    color = filt[:, :cascade_number],
    width = 0.05,
)

lm_fit = lm(@formula(community_niche_overlap ~ cascade_number), filt)


