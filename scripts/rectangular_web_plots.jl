using WGLMakie
using DataFrames
using CSV
using Statistics
using CategoricalArrays

df = CSV.read("sim-output/rectangular-web-2026-01-05/data.csv", DataFrame)

# We've already shown there to be only a small effect on small webs. We're not
# really interested in doing that again with rectangular webs, so we'll limit
# ourselves here to only the larger webs. 
# 
# Besides, the more extinction trials a web is subjected to, the more
# un-rectangular it'll become. We don't want to undermine the premise of the
# analyses we're doing here by including small web trials, which would have
# become quite un-rectangular.
filter!(:richness_pre => x-> x >= 15, df)
# Some (none in most runs) of the simulations end early because of instability or
# something. We can just exclude those to be safe. Including them or not didn't
# change any results. 
filter!(:retcode => x -> x == "Success", df)

# Add a column for proportion of community gone extinct after a primary
# extinction. This is to account for differences in community size at the time
# of primary extinction. Early on, I did some sensitivity analyses to see if
# this would be problematic. Using raw number of secondary extinctions produced
# qualitatively the same results.
transform!(df, 
    [:richness_pre, :secondary_extinctions] => 
    (x, y) =  y ./ x 
    => :extinction_proportion
)

# A bunch of our analyses are on the cascade size distribution, so we're not
# interested extinction trials that didn't produce any secondary extinctions.
df_no_zero = filter(:secondary_extinctions => !iszero, df)

# ----------------------------------------------------------------------------
# Panel 1 
# Probability of at least one secondary extinction occuring versus the rate of
# adaptive foraging.
#
# Panel 2
# Expected cascade size (when they do occur) given as a proportion of the community
# that goes extinct.
# ----------------------------------------------------------------------------

# Probability of at least one extinction occuring
gd = groupby(df, :g)
f(x) = count(!iszero, x) / length(x)
p_extinction = combine(gd, 
    :extinction_proportion => f => :prob_extinction
)

# Expected number of secondary extinctions given that at least one occurs.
gd_no_zero = groupby(df_no_zero, :g)
expected_cascade_size = combine(gd_no_zero, 
    :extinction_proportion => mean => :expected_cascade_size
)

fig = Figure(size = (1000, 500)) 
panel1 = Axis(fig[1,1], 
    xlabel = "Rate of foraging adaptation",
    ylabel = "Probability of at least one secondary extinction",
    title = "Probability of extinction",
    xticks = unique(df[:, :g]),
    xtickformat = "{:.2f}"
)
lines!(panel1, 
    p_extinction[:, :g], 
    p_extinction[:, :prob_extinction],
    color = :black
)

panel2 = Axis(fig[1,2], 
    xlabel = "Rate of foraging adaptation",
    ylabel = "Expected proportion of community extinctions",
    title = "Expected proportion of community extinct",
    xticks = unique(df[:, :g]),
    xtickformat = "{:.2f}"
)
sort!(expected_cascade_size, :g)
empty!(panel2)
lines!(panel2, 
    expected_cascade_size[:, :g], 
    expected_cascade_size[:, :expected_cascade_size],
    color = :black
)

save("figures/rectangular_web_prob_expected_proportion_g.png", fig)

