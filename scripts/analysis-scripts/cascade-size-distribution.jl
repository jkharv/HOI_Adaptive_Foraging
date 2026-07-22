include("../../src/HOI_Adaptive_Foraging.jl")
using .HOI_Adaptive_Foraging

using WGLMakie
using SpeciesInteractionNetworks
using DataFrames
using CSV
using GLM
using Statistics
using CategoricalArrays

df1 = CSV.read("sim-output/one-extinction-2026-04-30/data.csv", DataFrame)
df4 = CSV.read("sim-output/four-extinctions-2026-04-30/data.csv", DataFrame)

df = vcat(df1, df4)

preprocessing!(df)

const HIGH_RICHNESS = 1 # :lightblue
const LOW_RICHNESS = 2  # :goldenrod
df.richness_treatment = map(
    x -> x >= 20 ? HIGH_RICHNESS : LOW_RICHNESS, df.richness_pre
)
df.cascade_occured = map(x -> x > 0 ? true : false, df.extinction_proportion)

#
# Probability of a cascade occuring
#

gdf = groupby(df, [:richness_treatment, :g])

gdf = combine(gdf, :extinction_proportion => 
    (x -> count(x -> x > 0, x) / length(x))
    => :cascade_probability
)

fig = Figure(size = (1200, 800))
ax  = Axis(fig[1,1], 
    xlabel = "Adaptation Rate", 
    ylabel = "Probability of Cascade",
    xticks = unique(df[:, :g]),
    xtickformat = "{:.2f}", 
)

low = filter(:richness_treatment => x -> x == LOW_RICHNESS, gdf)
lines!(ax, low[:, :g], low[:, :cascade_probability];
    color = :goldenrod, 
    linewidth = 5
)

high = filter(:richness_treatment => x -> x == HIGH_RICHNESS, gdf)
lines!(ax, high[:, :g], high[:, :cascade_probability]; 
    color = :lightblue, 
    linewidth = 5
)

# Now plot some fitted lines

filt = filter(:richness_treatment => x -> x == HIGH_RICHNESS, df)
rg = glm(@formula(cascade_occured ~ g), filt, Binomial(), LogitLink())
gs = collect(0:0.01:0.5)
ys = predict(rg, DataFrame(g = gs))

lines!(ax, gs, ys; 
    linestyle = :dot, 
    color = :lightblue,
    linewidth = 5,
)

filt = filter(:richness_treatment => x -> x == LOW_RICHNESS, df)
rg = glm(@formula(cascade_occured ~ g), filt, Binomial(), LogitLink())
gs = collect(0:0.01:0.5)
ys = predict(rg, DataFrame(g = gs))

lines!(ax, gs, ys; 
    linestyle = :dot, 
    color = :goldenrod,
    linewidth = 5
)

# ------------------------------------------------------------------------------- #
# Expected proportion of the community going extinct given that a cascade occurs. #
# ------------------------------------------------------------------------------- #

ax  = Axis(fig[1,2], 
    xlabel = "Adaptation Rate", 
    ylabel = "Cascade Size",
    xticks = unique(df[:, :g]),
    xtickformat = "{:.2f}", 
)

boxplot!(ax, 
    df[:, :g], 
    df[:, :extinction_proportion];
    dodge = df[:, :richness_treatment],
    color = map(x-> x == 1 ? :lightblue : :goldenrod, df.richness_treatment),
    width = 0.05,
    transparency = 0.9
)

for p in [0.5, 0.9, 0.99]

    x = combine( 
        groupby(filter(:richness_treatment => x-> x == HIGH_RICHNESS, df), :g), 
        :extinction_proportion => (x -> quantile(x, p)) => :q
    )
    sort!(x, :g)
    lines!(ax, x.g, x.q, linewidth = 5, color = :lightblue)

    x = combine( 
        groupby(filter(:richness_treatment => x-> x == LOW_RICHNESS, df), :g), 
        :extinction_proportion => (x -> quantile(x, p)) => :q
    )
    sort!(x, :g)
    lines!(ax, x.g, x.q, linewidth = 5, linestyle = :solid, color = :goldenrod)

end

save("figures/cascade-size-distribution-and-probability.png", fig)