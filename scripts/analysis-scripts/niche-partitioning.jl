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

df1  = CSV.read("sim-output/one-extinction-2026-09-15.csv", DataFrame)
jld1 = jldopen("sim-output/one-extinction-2026-09-15.jld2")
df1.dataset_num .= 1
df2  = CSV.read("sim-output/one-extinction-2026-10-30.csv", DataFrame)
jld2 = jldopen("sim-output/one-extinction-2026-10-30.jld2")
df2.dataset_num .= 2

df_disturbances = CSV.read("sim-output/disturbances-2026-09-02.csv", DataFrame)
jld_disturbances = jldopen("sim-output/disturbances-2026-09-02.jld2")

df_no_disturbances = CSV.read("sim-output/no-disturbances-2026-09-02.csv", DataFrame)
jld_no_disturbances = jldopen("sim-output/no-disturbances-2026-09-02.jld2")

preprocessing!(df1, jld1)
preprocessing!(df2, jld2)

preprocessing!(df_disturbances, jld_disturbances)
preprocessing!(df_no_disturbances, jld_no_disturbances)

df_ext = vcat(df1, df2)
df_dist = vcat(df_disturbances, df_no_disturbances)

df_disturbances.removal_fraction .= 0.9
df_no_disturbances.removal_fraction .= 0.0
df_ext.removal_fraction .= 1.0

filter!(:realized_web => !ismissing, df_ext)
filter!(:density_pre => !ismissing, df_ext)

filter!(:realized_web => !ismissing, df_dist)
filter!(:density_pre => !ismissing, df_dist)

select!(df_ext, 
    [:alpha_web, :cascade, :density_pre, 
     :realized_prob, :g, :extinction_proportion,
     :cascade_number, :removal_fraction]
)

select!(df_dist, [:alpha_web, :cascade, :density_pre, 
     :realized_prob, :g, :extinction_proportion,
     :cascade_number, :removal_fraction])

df = vcat(df_ext, df_dist)

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

transform!(df, [:alpha_web] =>
    ByRow((median ∘ values ∘ specificity))
    => :median_specificity
)

# --------------------------------------------------------------------------- #
# Do communities with adaptive foraging have less niche overlap?              #
# Yes                                                                         #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
# Do communities experience cascades differently depending on niche overlap?  #
# Yes, low overlap -> larger cascades                                         #
# --------------------------------------------------------------------------- #

filt = copy(df)
filter!(:cascade_number => x -> x < 5, filt)

fig = Figure(size = (1200, 600))
ax  = Axis(fig[1,1], 
    xlabel = "Adaptation Rate", 
    ylabel = "Community Niche Overlap"
)

boxplot!(ax, 
    filt[:, :g],
    filt[:, :community_niche_overlap];
    width = 0.05,
)

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

fig = Figure(size = (600,600))
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

# ------------------------------------------------------------ #
# Does niche partioning correspond to lower median specificity #
# ------------------------------------------------------------ #

filt = copy(df)
filter!(:cascade_number => x -> x == 1, filt)

fig = Figure(size = (600,600))
ax  = Axis(fig[1,1], xlabel = "Niche Overlap", ylabel = "Median Specificity")

scatter!(ax, filt[:, :community_niche_overlap], filt[:, :median_specificity])

save("figures/median-specificity-v-niche-overlap.png", fig)