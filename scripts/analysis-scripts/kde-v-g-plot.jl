include("../../src/HOI_Adaptive_Foraging.jl")
using .HOI_Adaptive_Foraging

using CairoMakie
using DataFrames
using CSV
using Statistics
using CategoricalArrays
using Distributions

df = CSV.read("sim-output/niche-model-2026-03-16/data.csv", DataFrame)

preprocessing!(df)

# Density (KDE) plots
fig = Figure(size = (750, 500))
ax  = Axis(fig[1,1], xlabel = "log(Cascade Size)", ylabel = "count")
for (i, g) in (enumerate ∘ groupby)(df, :g)

    density!(ax, 
        log.(g[:, :extinction_proportion]), 
        color = (:slategray, 0.4),
        offset = i,
        # normalization = :pdf
        
    )
end
save("./figures/kde-v-g.svg", fig)

# Fit exp scale parameter for each g value.
param = []
for (i, g) in (enumerate ∘ groupby)(df, :g)
    x = fit(Exponential, g[:, :extinction_proportion])
    append!(param, params(x))
end
    
# Rate parameter v g
fig = Figure(size = (750, 500))
ax  = Axis(fig[1,1], xlabel = "g", ylabel = "Exponential Scale Parameter")
gs = unique(df[:, :g])
lines!(ax, gs, param)
save("./figures/exp-scale-v-g.svg", fig)
