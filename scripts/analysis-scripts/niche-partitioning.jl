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

df1  = CSV.read("sim-output/1-extinction-2026-07-30.csv", DataFrame)
jld1 = jldopen("sim-output/1-extinction-2026-07-30.jld2")

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

# --------------------------------------------------------------------------- #
# Do communities with adaptive foraging have less niche overlap?              #
# Yes                                                                         #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Do communities experience cascades differently depending on niche overlap?  #
# Yes, low overlap -> larger cascades                                         #
# --------------------------------------------------------------------------- #

filt = copy(df)
filter!(:cascade_number => x -> x < 4, filt)
filter!(:extinction_proportion => x -> x > 0, filt)

fig = Figure(size = (1200, 800))
ax  = Axis(fig[1,1], 
    xlabel = "Adaptation Rate", 
    ylabel = "Community Niche Overlap"
)

boxplot!(ax, 
    filt[:, :g],
    filt[:, :community_niche_overlap];
    width = 0.05,
    dodge = filt[:, :cascade_number],
    color = filt[:, :cascade_number]
)

filt.foodweb_number = categorical(filt.foodweb_number)
lmfit = lm(@formula(community_niche_overlap ~ g ), filt)
ablines!(ax, coef(lmfit)[1], coef(lmfit)[2], linewidth = 4, color = :red)

filt = copy(df)
filter!(:cascade_number => x -> x < 4, filt)
filter!(:extinction_proportion => x -> x > 0, filt)

ax2  = Axis(fig[1,2], xlabel = "Cascade Size", )
empty!(ax2)
scatter!(ax2, 
    filt[:, :extinction_proportion],
    filt[:, :community_niche_overlap]
)

filt.foodweb_number = categorical(filt.foodweb_number)
lmfit = lm(@formula(community_niche_overlap ~ extinction_proportion), filt)
ablines!(ax2, coef(lmfit)[1], coef(lmfit)[2], linewidth = 4, color = :red)

save("figures/niche_overlap.png", fig)

# --------------------------------------------------------------- #
# Are cascades more common in communities with low niche overlap? #
# Yes                                                             #
# --------------------------------------------------------------- #

filt = copy(df)
filter!(:cascade_number => x -> x == 1, filt)
filt.cascade_occured = map(x -> x > 0 ? true : false, filt.extinction_proportion)

fig = Figure(size = (800,800))
ax  = Axis(fig[1,1],
    xlabel = "Community Niche Overlap",
    ylabel = "Cascade Probability"
)

scatter!(ax, filt.community_niche_overlap, filt.cascade_occured)

logit_fit = glm(@formula(cascade_occured ~ community_niche_overlap), filt, Binomial())
xs = collect(0.2:0.01:0.7)
ys = predict(logit_fit, DataFrame(community_niche_overlap = xs))

lines!(ax, xs, ys)

save("figures/niche_overlap_logit.png", fig)

# ------------------------------------------------------- #
# Do adaptive communities have greater niche partitioning #
# ------------------------------------------------------- #

filt = copy(df)
filter!(:cascade_number => x -> x < 5, filt)
filter!(:cascade_niche_breadth => (x -> x < 1.0), filt)
filter!(:community_niche_breadth => (x -> x < 1.0), filt)

fig = Figure(size = (1200, 800))
ax  = Axis(fig[1,1], 
    xlabel = "Adaptation Rate", 
    ylabel = "Community Averaged Niche Breadth"
)
boxplot!(ax, 
    filt[:, :g],
    filt[:, :community_niche_breath],
    dodge = filt[:, :cascade_number],
    color = filt[:, :cascade_number],
    width = 0.05
)

save("figures/niche_breadth.png", fig)