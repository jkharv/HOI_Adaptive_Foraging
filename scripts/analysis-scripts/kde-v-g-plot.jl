include("../../src/HOI_Adaptive_Foraging.jl")
using .HOI_Adaptive_Foraging

using WGLMakie
using DataFrames
using Statistics
using CSV
using Statistics
using CategoricalArrays
using Distributions

df1 = CSV.read("sim-output/one-extinction-2026-04-30/data.csv", DataFrame)
df2 = CSV.read("sim-output/two-extinctions-2026-04-30/data.csv", DataFrame)
df3 = CSV.read("sim-output/three-extinctions-2026-04-30/data.csv", DataFrame)
df4 = CSV.read("sim-output/four-extinctions-2026-04-30/data.csv", DataFrame)

df = vcat(df1, df2, df3, df4)

preprocessing!(df)

fig = Figure(size = (1000, 750))
ax  = Axis(fig[1,1], xlabel = "g", ylabel = "Exponential Scale Parameter")
gs = unique(df[:, :g])

function conf_fit(vec::Vector{Float64}, p)

    subsamples = zeros(Float64, 10_000)

    for i in eachindex(subsamples)

        sub = filter(x -> rand() > 0.3, vec) 
        subsamples[i] = (params ∘ fit)(Exponential, sub)[1]
    end

    lower = quantile(vec, 1-p)
    upper = quantile(vec, p)
    full = (params ∘ fit)(Exponential, vec)[1]

    return (lower, full, upper)
end

empty!(ax)
# Fit exp scale parameter for each g value.
estimates = []
for (i, g) in (enumerate ∘ groupby)(df, :g)
    x = conf_fit(g[:, :extinction_proportion], 0.60)
    push!(estimates, x)
end

lines!(ax, gs, [x[2] for x in estimates])

# Rate parameter v g
save("./figures/exp-scale-v-g.svg", fig)