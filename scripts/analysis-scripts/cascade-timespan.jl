include("../../src/HOI_Adaptive_Foraging.jl")
using .HOI_Adaptive_Foraging

using WGLMakie
using DataFrames
using CSV
using Statistics
using CategoricalArrays

df = CSV.read("sim-output/niche-model-2026-03-16/data.csv", DataFrame)

preprocessing!(df)

# We've already shown there to be only a small effect on small webs. We're not
# really interested in doing that again with rectangular webs, so we'll limit
# ourselves here to only the larger webs. 
filter!(:richness_pre => x-> x >= 20, df)

# Only interested in cascades that actually happen. And have a timespan of
# greater than zero. Lots of cascades have only a single species going
# secondarily extinct, which leads to inflated amount of zeros and detracts from
# the interesting features of the distribution.
filter!(:cascade_timespan => !isnan, df)

# Again as in previous plots, there are soo many points that a boxplot is not
# really readible. There are so many outlier points the entire thing is super
# messy. We'll plot lines for various percentiles to get an idea of the
# distribution.
gdf = combine(groupby(df, :g), 
    :cascade_timespan => (x -> quantile(x, 0.5))  => :q50,
    :cascade_timespan => (x -> quantile(x, 0.75)) => :q75,
    :cascade_timespan => (x -> quantile(x, 0.9))  => :q90,
    :cascade_timespan => (x -> quantile(x, 0.95)) => :q95,
    :cascade_timespan => (x -> quantile(x, 0.99)) => :q99,
    :cascade_timespan => (x -> count(iszero, x) / length(x)) => :prop_zero
)
sort!(gdf, :g)

# Panel 1:
# Percentile lines showing how the distribution of cascade timespans changes
# with g.
fig = Figure(size = (1_000, 500))
ax  = Axis(fig[1,1:2], xlabel = "g", ylabel = "Cascade timespan")
xlims!(ax, 0, 0.5)
legend_entries = []
for l in names(gdf, r"q.")
    push!(legend_entries, lines!(ax, gdf[:, :g], gdf[:, l]))
end

# Panel 2:
# Proportion of cascades which only result in 1 species going extinct (not
# counting the primary extinction).
ax = Axis(fig[1,3], xlabel = "g", ylabel = "Proportion of size 1 cascades")
xlims!(ax, 0, 0.5)
lines!(ax, gdf[:, :g], gdf[:, :prop_zero], color = :black)

Legend(fig[1,3][2,2], 
    legend_entries,
    [L"50^{th} %", L"75^{th} %", L"90^{th} %", L"95^{th} %", L"99^{th} %"]
)

save("figures/cascade-timespan.svg", fig)