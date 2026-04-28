include("../../src/HOI_Adaptive_Foraging.jl")
using .HOI_Adaptive_Foraging

using WGLMakie
using DataFrames
using CSV
using GLM
using Statistics
using CategoricalArrays
using QuantileRegressions

function boundary(sptl::Float64, s::Float64)::Float64

    return s / sptl
end

# One extinction
df1 = CSV.read("sim-output/new-tl-algorithm-2026-04-20/data.csv", DataFrame)
# Two extinctions
df2 = CSV.read("sim-output/two-extinctions-2026-04-20/data.csv", DataFrame)

preprocessing!(df1)
preprocessing!(df2)

# Trophic range as a proportion of community richness. Once I start collecting
# info in the simulations, this should probably be a propotion of the max
# trophic level in the realized web.
[df1, df2] .|> x -> transform!(x,
    [:maximum_trophic_level, :trophic_range] =>
    ((x, y) ->  y ./ x)
    => :trophic_scale
);

[df1, df2] .|> x -> transform!(x,
    [:secondary_extinctions, :trophic_range] =>
    ((x, y) ->  x ./ y)
    => :extinctions_per_trophic_level
);

[df1, df2] .|> x -> transform!(x,
    [:richness_pre, :maximum_trophic_level] =>
    ((x, y) ->  x ./ y)
    => :spp_per_trophic_level
);

[df1, df2] .|> x -> transform!(x,
    [:trophic_mean, :target_trophic_level] =>
    ((x, y) ->  y - x)
    => :trophic_difference
);

# -------------------------------------------------------- #
# Trophic Difference Analyses:                             #
# Are top-down or bottom-up effects implicated more often? #
# Positive is top-down, negative is bottom-up              #
# -------------------------------------------------------- #

filt = vcat(df1, df2)
filter!(:trophic_difference => !ismissing, filt)
disallowmissing!(filt, :trophic_difference)

f  = Figure(size = (800, 800))
ax = Axis(f[1,1], xlabel = "g", ylabel = "Trophic Difference") 

boxplot!(ax, 
    filt[:, :g], 
    filt[:, :trophic_difference];
    dodge = filt[:, :n_targets],
    color = map(d-> d == 1 ? :blue : :red, filt[:, :n_targets]),
    width = 0.05
)

save("figures/trophic_difference_v_g.png", f)

# I made a graph of proportion extinct v. trophic difference but there is no
# discernable difference between large and small cascades wrt wether top-down or
# bottom-up are implicated.