include("../../src/HOI_Adaptive_Foraging.jl")
using .HOI_Adaptive_Foraging

using CairoMakie
using SpeciesInteractionNetworks
using DataFrames
using CSV
using Statistics
using CategoricalArrays

df_low  = CSV.read("sim-output/low-connectence-2026-05-31/data.csv", DataFrame)
df_high = CSV.read("sim-output/high-connectance-2026-05-31/data.csv", DataFrame)

df_low.connectance .= :low
df_high.connectance .= :high

df = vcat(df_low, df_high)

preprocessing!(df)

# We've already shown there to be only a small effect on small webs. We're not
# really interested in doing that again, so we'll limit ourselves here to only
# the larger webs.
filter!(:richness_pre => x-> x >= 20, df)

# --------------------------------------------- #
# Probability of a cascade of any size occuring #
# --------------------------------------------- #

gdf = groupby(df, [:connectance, :g])

p_extinction = combine(gdf,
    :extinction_proportion => 
    (x -> count(!iszero, x) / length(x))
    => :p_extinction
);

fig = Figure(size = (1000, 500))
ax  = Axis(fig[1,1], xlabel = "g", ylabel = "P(cascade)")
for n in unique(df[:, :connectance])

    colour = (n == :low) ? :gray : :red
    filt = filter(:connectance => x-> x == n, p_extinction)
    sort!(filt, :g)
    lines!(ax, filt[:, :g], filt[:, :p_extinction], color = colour)
end

# ------------------------------------------------------------------------------- #
# Expected proportion of the community going extinct given that a cascade occurs. #
# ------------------------------------------------------------------------------- #

filt = filter(:secondary_extinctions => !iszero, df)

colours = map(filt[:, :connectance]) do x

    if x == :low
        return :gray
    elseif x == :high
        return :red 
    end
end

dodge = map(filt[:, :connectance]) do x

    if x == :low
        return 1 
    elseif x == :high
        return 2
    end
end

ax  = Axis(fig[1,2], 
    xlabel = "Adaptation Rate", 
    ylabel = "Cascade Size",
    xticks = unique(filt[:, :g]),
    xtickformat = "{:.2f}" 
)

boxplot!(ax, 
    filt[:, :g], 
    filt[:, :extinction_proportion];
    dodge = dodge,
    color = colours,
    width = 0.05,
)

save("figures/prop-extinction-cascade-size-connectance.svg", fig)