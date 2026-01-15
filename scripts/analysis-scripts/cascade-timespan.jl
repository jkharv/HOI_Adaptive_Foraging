using WGLMakie
using DataFrames
using CSV
using Statistics
using CategoricalArrays

df = CSV.read("sim-output/niche-model-2026-01-08a/data.csv", DataFrame)

# We've already shown there to be only a small effect on small webs. We're not
# really interested in doing that again with rectangular webs, so we'll limit
# ourselves here to only the larger webs. 
#
# Besides, the more extinction trials a web is subjected to, the more
# un-rectangular it'll become. We don't want to undermine the premise of the
# analyses we're doing here by including small web trials, which would have
# become quite un-rectangular.
filter!(:richness_pre => x-> x >= 5, df)

# Some (none in most runs) of the simulations end early because of instability
# or something. We can just exclude those to be safe. Including them or not
# didn't change any results. 
filter!(:retcode => x -> x == "Success", df)

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

names(df)