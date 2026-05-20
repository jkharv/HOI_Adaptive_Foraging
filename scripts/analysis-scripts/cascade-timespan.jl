include("../../src/HOI_Adaptive_Foraging.jl")
using .HOI_Adaptive_Foraging

using WGLMakie
using DataFrames
using CSV
using Statistics
using CategoricalArrays
using GLM

# One extinction
df1 = CSV.read("sim-output/new-tl-algorithm-2026-04-20/data.csv", DataFrame)
# Two extinctions
df2 = CSV.read("sim-output/two-extinctions-2026-04-20/data.csv", DataFrame)
# Three extinctions
df3 = CSV.read("sim-output/three-extinctions-2026-04-23/data.csv", DataFrame)
# Four extinctions
df4 = CSV.read("sim-output/three-extinctions-2026-04-23/data.csv", DataFrame)

preprocessing!(df1)
preprocessing!(df2)
preprocessing!(df3)
preprocessing!(df4)

# foodweb_number and sequence_number are made non-unique by doing this.
df = vcat(df1, df2, df3, df4)

# We've already shown there to be only a small effect on small webs. We're not
# really interested in doing that again with rectangular webs, so we'll limit
# ourselves here to only the larger webs. 
filter!(:richness_pre => x-> x >= 20, df)

filter!(:secondary_extinctions => !iszero, df)

# We'll make a new column which is unique up to g
transform!(df, [:foodweb_number, :sequence_number, :n_targets] =>
    ByRow((x,y,z) ->
        string(x) * string(y) * string(z)
    ) => :id
)

gdf = groupby(df, :id)

lms = DataFrame()
for group in gdf

    slope = coef(lm(@formula(cascade_timespan ~ g), group))[2]
    push!(lms, (
        n_targets = group[:, :n_targets][1],
        slope = slope 
    )
    )
end

colours = map(lms[:, :n_targets]) do x

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
    ylabel = "Cascade Timespan LM",
    xticks = unique(lms[:, :n_targets]),
    xtickformat = "{:2d}"
)
boxplot!(ax, 
    lms[:, :n_targets], 
    lms[:, :slope];
    # dodge = filt[:, :n_targets],
    # color = filt[:, :g],
    # width = 0.05,
)

lm(@formula(cascade_timespan ~ g), df)