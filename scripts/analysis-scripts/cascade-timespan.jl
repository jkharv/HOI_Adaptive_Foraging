using CairoMakie
using DataFrames
using CSV
using Statistics
using CategoricalArrays

df = CSV.read("sim-output/niche-model-2026-01-12/data.csv", DataFrame)

# We've already shown there to be only a small effect on small webs. We're not
# really interested in doing that again with rectangular webs, so we'll limit
# ourselves here to only the larger webs. 
#
# Besides, the more extinction trials a web is subjected to, the more
# un-rectangular it'll become. We don't want to undermine the premise of the
# analyzes we're doing here by including small web trials, which would have
# become quite un-rectangular.
filter!(:richness_pre => x-> x >= 20, df)

# Some (none in most runs) of the simulations end early because of instability
# or something. We can just exclude those to be safe. Including them or not
# didn't change any results. 
filter!(:retcode => x -> x == "Success", df)

# Only interested in cascades that actually happen. And have a timespan of
# greater than zero. Lots of cascades have only a single species going
# secondarily extinct, which leads to inflated amount of zeros and detracts from
# the interesting features of the distribution.
filter!(:timespan_of_cascade => !isnan, df)
# filter!(:timespan_of_cascade => (x-> x>0), df)

# Add a column for proportion of community gone extinct after a primary
# extinction. This is to account for differences in community size at the time
# of primary extinction. Early on, I did some sensitivity analyses to see if
# this would be problematic. Using raw number of secondary extinctions produced
# qualitatively the same results.
transform!(df,
    [:richness_pre, :secondary_extinctions] =>
    ((x, y) ->  y ./ x)
    => :extinction_proportion
)

# We'll normalize the timespans since raw timesteps don't have any intuitive
# meaning. 0 - 1 where 1 is the maximum timepspan
max_t = maximum(df[:, :timespan_of_cascade])
transform!(df,
    :timespan_of_cascade =>
    (x ->  x ./ max_t)
    => :timespan_of_cascade
)

# Again as in previous plots, there are soo many points that a boxplot is not
# really readible. There are so many outlier points the entire thing is super
# messy. We'll plot lines for various percentiles to get an idea of the
# distribution.
gdf = combine(groupby(df, :g), 
    :timespan_of_cascade => (x -> quantile(x, 0.5))  => :q50,
    :timespan_of_cascade => (x -> quantile(x, 0.75)) => :q75,
    :timespan_of_cascade => (x -> quantile(x, 0.9))  => :q90,
    :timespan_of_cascade => (x -> quantile(x, 0.95))  => :q95,
    :timespan_of_cascade => (x -> quantile(x, 0.99)) => :q99,
    :timespan_of_cascade => (x -> count(iszero, x) / length(x)) => :prop_zero
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