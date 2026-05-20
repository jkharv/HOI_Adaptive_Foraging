include("../../src/HOI_Adaptive_Foraging.jl")
using .HOI_Adaptive_Foraging

using WGLMakie
using DataFrames
using CSV
using GLM
using Statistics
using SpeciesInteractionNetworks 
using HigherOrderFoodwebs

# I should redo this to be number of disturbances rather than number of
# extinctions

df1 = CSV.read("sim-output/one-extinction-2026-04-30/data.csv", DataFrame)
df2 = CSV.read("sim-output/two-extinctions-2026-04-30/data.csv", DataFrame)
df3 = CSV.read("sim-output/three-extinctions-2026-04-30/data.csv", DataFrame)
df4 = CSV.read("sim-output/four-extinctions-2026-04-30/data.csv", DataFrame)

df = vcat(df1, df2, df3, df4)

preprocessing!(df)

select!(df, [:n_targets, :foodweb_number, :sequence_number, :g, :richness_pre])
gdf = groupby(df, [:n_targets, :foodweb_number, :sequence_number, :g])

gdf

r50df = DataFrame()

for group in gdf

    sort!(group, :richness_pre, rev = true)

    initial_richness = group[1, :richness_pre]

    r50 = findfirst(x -> x <= ceil(initial_richness / 2), group[:, :richness_pre])
    # r50 = r50 * group[1, :n_targets]

    if isnothing(r50)

        continue 
    end

    push!(r50df, (
        foodweb_number = group[1, :foodweb_number],
        sequence_number = group[1, :sequence_number],
        n_targets = group[1, :n_targets],
        g = group[1, :g],
        r50 = r50
        )
    )
end

colours = map(r50df[:, :n_targets]) do x

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

f = Figure(size = (800, 800))
ax = Axis(f[1,1], 
    xlabel = "g", 
    ylabel = "R50",
    yticks = collect(1:15),
    xticks = unique(r50df[:, :g]) ,
    xtickformat = "{:.2f}"
)
boxplot!(ax, r50df[:, :g], r50df[:, :r50]; 
    width = 0.05,
    dodge =  r50df[:, :n_targets],
    color = colours,
)

save("figures/r50.png", f)