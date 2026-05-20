include("../../src/HOI_Adaptive_Foraging.jl")
using .HOI_Adaptive_Foraging

using WGLMakie
using SpeciesInteractionNetworks
using DataFrames
using CSV
using Statistics
using CategoricalArrays

# One extinction
df1 = CSV.read("sim-output/one-extinction-2026-04-30/data.csv", DataFrame)
# Two extinctions
df2 = CSV.read("sim-output/two-extinctions-2026-04-30/data.csv", DataFrame)
# Three extinctions
df3 = CSV.read("sim-output/three-extinctions-2026-04-30/data.csv", DataFrame)
# Four extinctions
df4 = CSV.read("sim-output/four-extinctions-2026-04-30/data.csv", DataFrame)

df = vcat(df1, df2, df3, df4)

preprocessing!(df)

# We've already shown there to be only a small effect on small webs. We're not
# really interested in doing that again with rectangular webs, so we'll limit
# ourselves here to only the larger webs.
filter!(:richness_pre => x-> x >= 20, df)

# --------------------------------------------- #
# Probability of a cascade of any size occuring #
# --------------------------------------------- #

gdf = groupby(df, [:n_targets, :g])

p_extinction = combine(gdf,
    :extinction_proportion => 
    (x -> count(!iszero, x) / length(x))
    => :p_extinction
);

fig = Figure(size = (1000, 500))
ax  = Axis(fig[1,1], xlabel = "g", ylabel = "P(cascade)")
for n in unique(df[:, :n_targets])

    filt = filter(:n_targets => x-> x == n, p_extinction)
    sort!(filt, :g)
    lines!(ax, filt[:, :g], filt[:, :p_extinction])
end

ylims!(ax, [0.8, 1.0])

save("figures/prob-extinctions-disturbance-size.png", fig)

# ------------------------------------------------------------------------------- #
# Expected proportion of the community going extinct given that a cascade occurs. #
# ------------------------------------------------------------------------------- #

filt = filter(:secondary_extinctions => !iszero, df)

colours = map(filt[:, :n_targets]) do x

    if x == 1
        return :blue
    elseif x == 2
        return :green
    elseif x == 3
        return :red
    else
        return :grey
    end
end

fig = Figure(size = (1200, 750))
ax  = Axis(fig[1,1], 
    xlabel = "Adaptation Rate", 
    ylabel = "Cascade Size",
    xticks = unique(filt[:, :g]),
    xtickformat = "{:.2f}"
    
)

boxplot!(ax, 
    filt[:, :g], 
    filt[:, :extinction_proportion];
    dodge = filt[:, :n_targets],
    color = colours,
    width = 0.05,
)

save("figures/prop-extinction-v-disturbance-size.png", fig)
