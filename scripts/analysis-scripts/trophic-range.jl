using WGLMakie
using DataFrames
using CSV
using GLM
using Statistics
using CategoricalArrays
using QuantileRegressions

const RICHNESS_CUTOFF = 5

function boundary(sptl::Float64, s::Float64)::Float64

    return s / sptl
end

df = CSV.read("sim-output/niche-model-2026-02-10/data.csv", DataFrame)

filter!(:richness_pre => x-> x >= RICHNESS_CUTOFF, df)

# Some (none in most runs) of the simulations end early because of instability
# or something. We can just exclude those to be safe. Including them or not
# didn't change any results. 
filter!(:retcode => x -> x == "Success", df)

filter!(:secondary_extinctions => x-> x>0, df)

# There are some reallllyyyy small values in here causing extremely large values
# later on.  These are probably floating point error, we'll reassign these to
# zero.
transform!(df, 
    :trophic_range => ByRow(x -> (x<0.01) ? 0 : x) => :trophic_range    
)

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

# Trophic range as a proportion of community richness. Once I start collecting
# info in the simulations, this should probably be a propotion of the max trophic
# level in the realized web.
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
#
# Cascade size ~ Trophic range
#

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

#
#
#
#
#

filt = filter(:g => x -> x>-0.2, df)
filt = filter(:secondary_extinctions => x -> x>0, filt)

transform!(filt,
    [:maximum_trophic_level, :spp_per_trophic_level] =>
    ((x, y) ->  x ./ y)
    => :test
)

#jitter
xj = 0.05 
yj = 0.5

fig = Figure(size = (1600, 850))
ax1  = Axis(fig[1,1], 
    ylabel = "Number of Secondary Extinctions", 
    xlabel = "g",
    title  = ""
)
empty!(ax1)
scatter!(ax1, 
    filt[:, :maximum_trophic_level] + xj*rand(nrow(filt)),
    filt[:, :spp_per_trophic_level] + yj*rand(nrow(filt)),
    color = filt[:, :trophic_range],
    colormap = :bluesreds,
    markersize = 2*filt[:, :secondary_extinctions]
)