using WGLMakie
using DataFrames
using CSV
using GLM
using Statistics
using CategoricalArrays
using QuantileRegressions

df = CSV.read("sim-output/niche-model-2026-01-31/data.csv", DataFrame)

filter!(:richness_pre => x-> x >= 10, df)

# Some (none in most runs) of the simulations end early because of instability
# or something. We can just exclude those to be safe. Including them or not
# didn't change any results. 
filter!(:retcode => x -> x == "Success", df)

filter!(:secondary_extinctions => x-> x>0, df)

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
    [:richness_pre, :trophic_range] =>
    ((x, y) ->  y ./ x)
    => :trophic_scale
)

#
# Cascade size ~ Trophic range
#
filt = filter(:g => x -> x>0.1, df)
filt = filter(:trophic_scale => x -> x>0, filt)

fig = Figure()
ax  = Axis(fig[1,1], ylabel = "Cascade Size", xlabel = "Trophic Scale")
scatter!(ax, filt[:, :trophic_scale], filt[:, :extinction_proportion])

r = qreg(@formula(extinction_proportion ~ trophic_scale), filt, 0.95)

ablines!(ax, coef(r)[1], coef(r)[2])


names(df)