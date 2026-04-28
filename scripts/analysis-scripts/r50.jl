include("../../src/HOI_Adaptive_Foraging.jl")
using .HOI_Adaptive_Foraging

using WGLMakie
using DataFrames
using CSV
using GLM
using Statistics
using SpeciesInteractionNetworks 
using HigherOrderFoodwebs

df1 = CSV.read("sim-output/new-tl-algorithm-2026-04-20/data.csv", DataFrame)
df2 = CSV.read("sim-output/two-extinctions-2026-04-21/data.csv", DataFrame)

df = vcat(df1, df2)

preprocessing!(df)

select!(df, [:n_targets, :foodweb_number, :sequence_number, :g, :richness_pre])
gdf = groupby(df, [:n_targets, :foodweb_number, :sequence_number, :g])

r50df = DataFrame()

for group in gdf

    sort!(group, :richness_pre, rev = true)

    initial_richness = group[1, :richness_pre]

    r50 = findfirst(x -> x <= ceil(initial_richness / 2), group[:, :richness_pre])
    r50 = r50 * group[1, :n_targets]

    push!(r50df, (
        foodweb_number = group[1, :foodweb_number],
        sequence_number = group[1, :sequence_number],
        n_targets = group[1, :n_targets],
        g = group[1, :g],
        r50 = r50
        )
    )
end

f = Figure(size = (800, 800))
ax = Axis(f[1,1], xlabel = "g", ylabel = "R50")
boxplot!(ax, r50df[:, :g], r50df[:, :r50]; 
    width = 0.05,
    dodge =  r50df[:, :n_targets],
    color = map(d-> d == 1 ? :blue : :red, r50df[:, :n_targets]),

)

save("figures/r50.png", f)