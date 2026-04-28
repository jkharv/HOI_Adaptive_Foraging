include("../../src/HOI_Adaptive_Foraging.jl")
using .HOI_Adaptive_Foraging

using WGLMakie
using SpeciesInteractionNetworks
using DataFrames
using CSV
using Statistics
using CategoricalArrays

# One extinction
df1 = CSV.read("sim-output/new-tl-algorithm-2026-04-20/data.csv", DataFrame)
# Two extinctions
df2 = CSV.read("sim-output/two-extinctions-2026-04-20/data.csv", DataFrame)
# Three extinctions
df3 = CSV.read("sim-output/three-extinctions-2026-04-23/data.csv", DataFrame)

preprocessing!(df1)
preprocessing!(df2)
preprocessing!(df3)

# We've already shown there to be only a small effect on small webs. We're not
# really interested in doing that again with rectangular webs, so we'll limit
# ourselves here to only the larger webs.
filter!(:richness_pre => x-> x >= 20, df1)
filter!(:richness_pre => x-> x >= 20, df2)
filter!(:richness_pre => x-> x >= 20, df3)

# --------------------------------------------- #
# Probability of a cascade of any size occuring #
# --------------------------------------------- #

gdf1 = groupby(df1, :g)
gdf2 = groupby(df2, :g)
gdf3 = groupby(df3, :g)

f(x) = count(!iszero, x) / length(x)
p_extinction1 = combine(gdf1,
    :extinction_proportion => f => :p_extinction
);
p_extinction2 = combine(gdf2,
    :extinction_proportion => f => :p_extinction
);
p_extinction3 = combine(gdf3,
    :extinction_proportion => f => :p_extinction
);

fig = Figure(size = (1000, 500))
ax  = Axis(fig[1,1], xlabel = "g", ylabel = "P(cascade)")
lines!(ax, p_extinction1[:, :g], p_extinction1[:, :p_extinction]; color = :black)
lines!(ax, p_extinction2[:, :g], p_extinction2[:, :p_extinction]; color = :red)
lines!(ax, p_extinction3[:, :g], p_extinction3[:, :p_extinction]; color = :red)

save("figures/prob-extinctions-disturbance-size.png", fig)

# ------------------------------------------------------------------------------- #
# Expected proportion of the community going extinct given that a cascade occurs. #
# ------------------------------------------------------------------------------- #

filt1 = filter(:secondary_extinctions => !iszero, df1)
filt2 = filter(:secondary_extinctions => !iszero, df2)
filt3 = filter(:secondary_extinctions => !iszero, df3)
filt = vcat(filt1, filt2, filt3)

filter!(:richness_pre => x-> x>25, filt)

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
    xlabel = "g", 
    ylabel = "Proportion extinct",
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
