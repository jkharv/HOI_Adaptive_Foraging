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

df = CSV.read("sim-output/niche-model-2026-03-16/data.csv", DataFrame)

preprocessing!(df)

# Trophic range as a proportion of community richness. Once I start collecting
# info in the simulations, this should probably be a propotion of the max
# trophic level in the realized web.
transform!(df,
    [:maximum_trophic_level, :trophic_range] =>
    ((x, y) ->  y ./ x)
    => :trophic_scale
)

transform!(df,
    [:secondary_extinctions, :trophic_range] =>
    ((x, y) ->  x ./ y)
    => :extinctions_per_trophic_level
)

transform!(df,
    [:richness_pre, :maximum_trophic_level] =>
    ((x, y) ->  x ./ y)
    => :spp_per_trophic_level
)

transform!(df,
    [:trophic_mean, :target_trophic_level] =>
    ((x, y) ->  y - x)
    => :trophic_difference
)

# ----------------------------------------------------------------------------- #
# Food web Height v. Width Analyses:                                            #
# Do cascades happen at different rates in food webs which are short and fat or #
# tall and skinny? Does adaptive foraging affect these differently?             #
# ----------------------------------------------------------------------------- #

# Weak adaptive foraging
filt = filter(:g => x -> x<0.1, df)
filt = filter(:secondary_extinctions => x -> x>0, filt)

#jitter
j = 0.5 

fig = Figure(size = (1400, 750))
ax1  = Axis(fig[1,1], 
    ylabel = "Maximum Trophic Level - (Height)", 
    xlabel = "Species per Trophic Level - (Width)", 
    title  = "Weak adaptive foraging (g < 0.1)"
)
scatter!(ax1, 
    filt[:, :maximum_trophic_level] + j*rand(nrow(filt)),
    filt[:, :spp_per_trophic_level] + j*rand(nrow(filt)),
    color = log.(filt[:, :extinction_proportion]),
    colormap = :bluesreds,
    markersize = 40 * filt[:, :extinction_proportion]
)

xs = collect(0:0.01:10)
lines!(ax1, xs, boundary.(xs, 5.0))
lines!(ax1, xs, boundary.(xs, 30.0))

# Strong adaptive foraging.
filt = filter(:g => x -> x>0.2, df)
filt = filter(:secondary_extinctions => x -> x>0, filt)

ax2  = Axis(fig[1,2], 
    ylabel = "Maximum Trophic Level - (Height)", 
    xlabel = "Species per Trophic Level - (Width)", 
    title  = "Strong adaptive foraging (g > 0.2)"
)
scatter!(ax2, 
    filt[:, :maximum_trophic_level] + j*rand(nrow(filt)),
    filt[:, :spp_per_trophic_level] + j*rand(nrow(filt)),
    color = log.(filt[:, :extinction_proportion]),
    colormap = :bluesreds,
    markersize = 40 * filt[:, :extinction_proportion]
)

xs = collect(0:0.01:10)
lines!(ax2, xs, boundary.(xs, 5.0))
lines!(ax2, xs, boundary.(xs, 30.0))

xlims!(ax1, 0, 10)
xlims!(ax2, 0, 10)

ylims!(ax1, 0, 10)
ylims!(ax2, 0, 10)

save("figures/shape-v-cascades.png", fig)

# -------------------------------------------------------- #
# Trophic Difference Analyses:                             #
# Are top-down or bottom-up effects implicated more often? #
# -------------------------------------------------------- #

filt = copy(df)
filter!(:trophic_difference => !ismissing, filt)
disallowmissing!(filt, :trophic_difference)

f  = Figure(size = (650, 650))
ax = Axis(f[1,1], xlabel = "g", ylabel = "Trophic Difference") 

boxplot!(ax, 
    filt[:, :g], 
    filt[:, :trophic_difference];
    width = 0.05
)

save("figures/trophic_difference_v_g.png", f)

f  = Figure(size = (650, 650))
ax = Axis(f[1,1], 
    xlabel = "Trophic Difference", 
    ylabel = "Proportion Extinct"
) 
scatter!(ax, 
    filt[:, :trophic_difference], 
    filt[:, :extinction_proportion]
)

save("figures/trophic_difference_v_proportion_extinct.png", f)