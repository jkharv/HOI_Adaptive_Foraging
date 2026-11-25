include("../../src/HOI_Adaptive_Foraging.jl")

using .HOI_Adaptive_Foraging

using WGLMakie
using DataFrames
using CSV
using JLD2
using GLM
using StatsBase
using LsqFit
using SpeciesInteractionNetworks 
using DataStructures
using HigherOrderFoodwebs

df1  = CSV.read("sim-output/one-extinction-2026-09-15.csv", DataFrame)
jld1 = jldopen("sim-output/one-extinction-2026-09-15.jld2")
df1.dataset_num .= 1
df2  = CSV.read("sim-output/one-extinction-2026-10-30.csv", DataFrame)
jld2 = jldopen("sim-output/one-extinction-2026-10-30.jld2")
df2.dataset_num .= 2

preprocessing!(df1, jld1)
preprocessing!(df2, jld2)
df = vcat(df1, df2)

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

transform!(df,
    [:density_pre] =>
    ByRow(shannon_eveness) =>
    :eveness
)

# --------------------------------------------------------------------------- #
# Top-Left Panel
#

filt = copy(df)
filter!(:cascade_number => x -> x == 1, filt)

fig = Figure(size = (900, 900))
ax11  = Axis(fig[1,1], 
    ylabel = "Community Niche Overlap",
    xticks = unique(filt[:, :g]),
    xtickformat = "{:.2f}", 
)

boxplot!(ax11, 
    filt[:, :g],
    filt[:, :community_niche_overlap]; 
    width = 0.05,
)

lmfit = lm(@formula(community_niche_overlap ~ g ), filt)
ablines!(ax11, coef(lmfit)[1], coef(lmfit)[2], linewidth = 4, color = :red)

#
# Top-Right Panel
#

ax12  = Axis(fig[1,2])
scatter!(ax12, 
    filt[:, :extinction_proportion],
    filt[:, :community_niche_overlap]   
)

model(t, p) = p[1] .+ (p[2] .* t)
fitOLS = curve_fit(model, 
    filt[:, :extinction_proportion], 
    filt[:, :community_niche_overlap], 
    [1.0, 1.0]
)
wt = 1 ./ fitWLS.resid.^2
fitWLS = curve_fit(model, 
    filt[:, :extinction_proportion], 
    filt[:, :community_niche_overlap], 
    wt, 
    [0.0, 0.0]
)

ablines!(ax12, fitWLS.param[1], fitWLS.param[2], color=:red, linewidth = 3)

#
#  Bottom-Left Panel
#

ax21 = Axis(fig[2,1],
    xlabel = "Adaptation Rate",
    ylabel = "Shannon Eveness",
    xticks = unique(filt[:, :g]),
    xtickformat = "{:.2f}", 
)

boxplot!(ax21, 
    filt[:, :g],
    filt[:, :eveness],
    width = 0.05,
)

lmfit = lm(@formula(eveness ~ g), filt)
ablines!(ax21, coef(lmfit)[1], coef(lmfit)[2], linewidth = 4, color = :red)

#
# Bottom Right Panel
#

ax22 = Axis(fig[2,2],
    xlabel = "Cascade Size",
)

scatter!(ax22, 
    filt[:, :extinction_proportion],
    filt[:, :eveness],
)

model(t, p) = p[1] .+ (p[2] .* t)
fitOLS = curve_fit(model, 
    filt[:, :extinction_proportion], 
    filt[:, :eveness], 
    [1.0, 1.0]
)
wt = 1 ./ fitOLS.resid.^2
fitWLS = curve_fit(model, 
    filt[:, :extinction_proportion], 
    filt[:, :eveness], 
    wt, 
    [0.0, 0.0]
)

ablines!(ax22, fitWLS.param[1], fitWLS.param[2], color=:red, linewidth = 3)

save("figures/niche-partitioning-eveness-figure.svg", fig)