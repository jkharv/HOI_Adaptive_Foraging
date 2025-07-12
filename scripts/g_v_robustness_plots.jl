using CairoMakie
using DataFrames
using CSV
using GLM
using Statistics
using CategoricalArrays

df = CSV.read("data_narval_s15_8g.csv", DataFrame)

filter!(:richness_pre => x-> x >= 5, df)
# Some (very few) of the simulation end early because of instability or
# something. We can just exclude those to be safe. Including them or not didn't
# change any results. 
filter!(:retcode => x -> x == "Success", df)
# Add a column for proportion of community gone extinct after a primary extinction.
f(x, y) =  y ./ x
transform!(df, [:richness_pre, :secondary_extinctions] => f => :extinction_proportion)

# ------------------------------------------------------------------------------
# Calculating some derived values from the raw extinction trial data. This will
# be plotted later in the file.
# ------------------------------------------------------------------------------

df_all_richness = df
df_high_richness = filter(:richness_pre => x-> x >= 10, df)

gd_all_richness = groupby(df_all_richness, :g)
gd_high_richness = groupby(df_high_richness, :g)

# Probability of at least one extinction occuring
f(x) = count(!iszero, x) / length(x)
p_extinction_all_richness = combine(gd_all_richness, 
    :extinction_proportion => f => :prob_extinction_all_richness
)
p_extinction_high_richness = combine(gd_high_richness, 
    :extinction_proportion => f => :prob_extinction_high_richness
)

# Expected number of secondary extinctions given that at least one occurs.
df_all_richness_no_zero = filter(:secondary_extinctions => !iszero, df_all_richness)
df_high_richness_no_zero = filter(:secondary_extinctions => !iszero, df_high_richness)

gd_all_richness_no_zero = groupby(df_all_richness_no_zero, :g)
gd_high_richness_no_zero = groupby(df_high_richness_no_zero, :g)

exp_n_extinctions_all_richness = combine(gd_all_richness_no_zero, 
    :extinction_proportion => mean => :expected_secondary_extinctions_all_richness
)
exp_n_extinctions_high_richness = combine(gd_high_richness_no_zero, 
    :extinction_proportion => mean => :expected_secondary_extinctions_high_richness
)

# We can combine those into one dataframe to make our lives easier.
df_derived = leftjoin(p_extinction_all_richness, p_extinction_high_richness; on = :g)
leftjoin!(df_derived, exp_n_extinctions_all_richness; on = :g)
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
# One line for richnesses 5-15, another for richnesses 13-15.
# ----------------------------------------------------------------------------

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
lines!(panel1, 
    df_derived[:, :g], 
    df_derived[:, :prob_extinction_all_richness],
    color = :black
)

panel2 = Axis(fig[1,2], 
    xlabel = "Rate of foraging adaptation",
    ylabel = "Expected proportion of community extinctions",
    title = "Expected proportion of community extinct",
    xticks = unique(df[:, :g]),
    xtickformat = "{:.2f}"
)
high_richness_line = lines!(panel2, 
    df_derived[:, :g], 
    df_derived[:, :expected_secondary_extinctions_high_richness],
    color = :red
)
all_richness_line = lines!(panel2, 
    df_derived[:, :g], 
    df_derived[:, :expected_secondary_extinctions_all_richness],
    color = :black
)

Legend(fig[1,2][-1, 2], 
    [all_richness_line, high_richness_line], 
    ["All richness (5-15)", "High richness (13-15)"]
)

save("figures/g_v_prob_and_expectation.svg", fig)

# ------------------------------------------------------------------------------
# Panel 1
# g versus the proportion of the community that goes extinct, for all richnesses
# (5-15)
#
# Panel 2
# g versus the proportion of the community that goes extinct, for only high
# richnesses (13-15)
#
# In both of these panels, the zeros have been removed. I discuss the
# probability of at least one extinction occuring seperately from the expected
# number of secondary extinctions.
# ------------------------------------------------------------------------------

filt = filter(:extinction_proportion => x -> x > 0, df)

fig = Figure(size = (1000, 500)) 
panel1 = Axis(fig[1,1], 
    xlabel = "Rate of foraging adaptation",
    ylabel = "Proportion of community extinct",
    title = "All richnesses (5-15)",
    xticks = unique(df[:, :g]),
    xtickformat = "{:.2f}"
)
ylims!(panel1, [0.0, 0.65])
boxplot!(panel1, 
    filt[:, :g], 
    filt[:, :extinction_proportion];
    width = 0.05
)

filter!(:richness_pre => x -> x >= 13, filt)

panel2 = Axis(fig[1,2], 
    xlabel = "Rate of foraging adaptation",
    title = "High richnesses only (13-15)",
    xticks = unique(df[:, :g]),
    xtickformat = "{:.2f}"
)
ylims!(panel2, [0.0, 0.65])
boxplot!(panel2, 
    filt[:, :g], 
    filt[:, :extinction_proportion];
    width = 0.05,
)

save("figures/g_v_secondary_extinctions.svg", fig)

# ---------------------------------------------------------------------- 
# Coefficient of regression against g at multiple values of richness pre 
# ---------------------------------------------------------------------- 

richnesses = [ ]
g_coefs = [ ]

for i in (minimum(df[:, :richness_pre])):(maximum(df[:, :richness_pre]))

    filt = filter(:richness_pre => x -> x ∈ (i-1):(i+1), df)
    mm = lm(@formula(extinction_proportion ~ g), filt)

    g_coef = coef(mm)[2]

    push!(g_coefs, g_coef)
    push!(richnesses, i)
end

fig = Figure() 
ax = Axis(fig[1,1], 
    xlabel = "Richness before extinction (± 1)",
    ylabel = "Regression coefficient of g",
    title = "Effect of adaptive foraging in communities of different species richness"
)
lines!(ax, richnesses, g_coefs)
hlines!(ax, 0, color = :red)
save("figures/g_effect_v_richness_pre.png", fig)

# --------------------------------------------------------
# Linear model 
#

mm = lm(@formula(extinction_proportion ~  g * richness_pre), df)