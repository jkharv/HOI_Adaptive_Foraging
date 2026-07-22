include("../../src/HOI_Adaptive_Foraging.jl")
using .HOI_Adaptive_Foraging

using DataFrames
using CSV
using GLM
using StatsBase
using Distributions
import WGLMakie as Makie
using Statistics
using SpeciesInteractionNetworks 
using DataStructures
using HigherOrderFoodwebs

df = CSV.read("sim-output/one-extinction-2026-04-30/data.csv", DataFrame)

preprocessing!(df)

# filter!(:realized_web => !ismissing, df)
# filter!(:density_pre => !ismissing, df)
filter!(:richness_pre => x-> x >= 20, df)

function sample_network(
    web::SpeciesInteractionNetwork{Unipartite{T}, <:Interactions}, 
    pthreshold::Float64
    )::SpeciesInteractionNetwork{Unipartite{T}, Binary{Bool}} where T


    retained_intx = filter(p -> last(p) >= pthreshold, interactions(web))

    retained_spp = Set{Symbol}()
    for (s, o, p) in retained_intx

        push!(retained_spp, s)
        push!(retained_spp, o)
    end
    retained_spp = collect(retained_spp)

    edges = zeros(Bool, length(retained_spp), length(retained_spp))

    new_network = SpeciesInteractionNetwork(Unipartite(retained_spp), Binary(edges))

    for (s, o, p) in retained_intx

        new_network[s,o] = true
    end

    return new_network
end

gdf = groupby(df, [:foodweb_number, :sequence_number, :g])
pairs = DataFrame()
for g in gdf

    sort!(g, [:richness_pre], rev = true)

    if nrow(g) == 1 
        continue
    end

    push!(pairs,
        (
            pre = sample_network(g[1, :realized_trim], 0.0), 
            post = sample_network(g[2, :realized_trim], 0.0),
            extinction_proportion = g[1, :extinction_proportion],
            g = g[1, :g]
        ) 
    )
end

transform!(pairs, 
    [:pre, :post] => 
    ByRow((pre, post) -> betadiversity(βOS, pre, post))
    => :div_os
)

transform!(pairs, 
    :div_os => 
    ByRow(x -> sum(x))
    => :intx_sum
)

transform!(pairs, 
    [:div_os, :intx_sum] => 
    ByRow((os, total) -> os.shared / total)
    => :prop_shared
)

pairs.prop_shared

fig = Makie.Figure(size = (1200, 800)) 
ax  = Makie.Axis(fig[1,1]; 
    xlabel = "Cascade Size", 
    ylabel = "Proportion Shared Interactions"
)
Makie.scatter!(ax, pairs.extinction_proportion, pairs.prop_shared)
