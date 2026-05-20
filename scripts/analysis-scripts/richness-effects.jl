include("../../src/HOI_Adaptive_Foraging.jl")
using .HOI_Adaptive_Foraging

using WGLMakie
using DataFrames
using CSV
using Statistics

df1 = CSV.read("sim-output/one-extinction-2026-04-30/data.csv", DataFrame)
df2 = CSV.read("sim-output/one-extinction-2026-04-30/data.csv", DataFrame)
df3 = CSV.read("sim-output/one-extinction-2026-04-30/data.csv", DataFrame)
df4 = CSV.read("sim-output/one-extinction-2026-04-30/data.csv", DataFrame)

df = vcat(df1, df2, df3, df4)

preprocessing!(df)

# ------------------------------------------------------------------------------
# Calculating some derived values from the raw extinction trial data. This will
# be plotted later in the file.
# ------------------------------------------------------------------------------

df_low_richness = filter(:richness_pre => x-> x < 15, df)
df_high_richness = filter(:richness_pre => x-> x >= 15, df)

gd_low_richness = groupby(df_low_richness, :g)
gd_high_richness = groupby(df_high_richness, :g)

# Probability of at least one extinction occuring
f(x) = count(!iszero, x) / length(x)
p_extinction_low_richness = combine(gd_low_richness, 
    :extinction_proportion => f => :prob_extinction_low_richness
)
p_extinction_high_richness = combine(gd_high_richness, 
    :extinction_proportion => f => :prob_extinction_high_richness
)

# Expected number of secondary extinctions given that at least one occurs.
df_low_richness_no_zero = filter(:secondary_extinctions => !iszero, df_low_richness)
df_high_richness_no_zero = filter(:secondary_extinctions => !iszero, df_high_richness)

gd_low_richness_no_zero = groupby(df_low_richness_no_zero, :g)
gd_high_richness_no_zero = groupby(df_high_richness_no_zero, :g)

exp_n_extinctions_low_richness = combine(gd_low_richness_no_zero, 
    :extinction_proportion => mean => :expected_secondary_extinctions_low_richness
)
exp_n_extinctions_high_richness = combine(gd_high_richness_no_zero, 
    :extinction_proportion => mean => :expected_secondary_extinctions_high_richness
)

# We can combine those into one dataframe to make our lives easier.
df_derived = leftjoin(p_extinction_low_richness, p_extinction_high_richness; on = :g)
leftjoin!(df_derived, exp_n_extinctions_low_richness; on = :g)
leftjoin!(df_derived, exp_n_extinctions_high_richness; on = :g)

# ----------------------------------------------------------------------------
# Panel 1 
# Probability of at least one secondary extinction occuring versus the rate of
# adaptive foraging.
#
# Panel 2
# Expected proportion of community to go extinct given that at least one
# secondary extinction occurs.
#
# One version for low richness (< 15). One for high richness (≥ 15) 
# ----------------------------------------------------------------------------

fig = Figure(size = (1000, 500)) 
panel1 = Axis(fig[1,1], 
    xlabel = "Rate of Adaptation",
    ylabel = "Probability of at least one secondary extinction",
    title = "Probability of extinction",
    xticks = unique(df[:, :g]),
    xtickformat = "{:.2f}"
)
lines!(panel1, 
    df_derived[:, :g], 
    df_derived[:, :prob_extinction_low_richness],
    color = :black
)

panel2 = Axis(fig[1,2], 
    xlabel = "Rate of foraging adaptation",
    ylabel = "Cascade Size",
    title = "Expected proportion of community extinct",
    xticks = unique(df[:, :g]),
    xtickformat = "{:.2f}"
)
low_richness_line = lines!(panel2, 
    df_derived[:, :g], 
    df_derived[:, :expected_secondary_extinctions_low_richness],
    color = :black
)

save("figures/g_v_prob_and_expectation_low_richness.png", fig)

fig = Figure(size = (1000, 500)) 
panel1 = Axis(fig[1,1], 
    xlabel = "Rate of foraging adaptation",
    ylabel = "Probability of at least one secondary extinction",
    title = "Probability of extinction",
    xticks = unique(df[:, :g]),
    xtickformat = "{:.2f}"
)
lines!(panel1,
    df_derived[:, :g],
    df_derived[:, :prob_extinction_high_richness],
    color = :red
)

panel2 = Axis(fig[1,2], 
    xlabel = "Rate of foraging adaptation",
    ylabel = "Expected proportion of community extinctions",
    title = "Expected proportion of community extinct",
    xticks = unique(df[:, :g]),
    xtickformat = "{:.2f}"
)
lines!(panel2, 
    df_derived[:, :g], 
    df_derived[:, :expected_secondary_extinctions_high_richness],
    color = :red
)

save("figures/g_v_prob_and_expectation_high_richness.png", fig)