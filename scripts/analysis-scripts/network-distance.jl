include("../../src/HOI_Adaptive_Foraging.jl")
using .HOI_Adaptive_Foraging

using DataFrames
using CSV
using GLM
using StatsBase
using Statistics
using Distributions
import WGLMakie as Makie
using Statistics
using SpeciesInteractionNetworks 
using DataStructures
using HigherOrderFoodwebs

df = CSV.read("sim-output/1-extinctions-2026-07-07/data.csv", DataFrame)

preprocessing!(df)

# filter!(:realized_web => !ismissing, df)
# filter!(:density_pre => !ismissing, df)
filter!(:richness_pre => x-> x >= 20, df)

web = df.alpha_web[1]
u   = df.density_pre[1]
threshold = 0.1

function sample_web(
    web::SpeciesInteractionNetwork{Unipartite{Symbol}, Quantitative{Float64}},
    u::Vector{Float64},
    threshold::Float64
    )::SpeciesInteractionNetwork{Unipartite{Symbol}, Binary{Bool}}

    # Retained species
    indxs = findall(!iszero, u[1:richness(web)])
    spp_retained = species(web)[indxs]
    extinct_spp  = setdiff(species(web), spp_retained)

    intxs_retained = filter(
        x -> (x[1] ∉ extinct_spp) & (x[2] ∉ extinct_spp) & (x[3] >= threshold), 
        interactions(web)
    )

    adj_new = zeros(Bool, length(spp_retained), length(spp_retained)) 

    for (s, o, v) in intxs_retained

        sindx = findfirst(x-> x == s, spp_retained)
        oindx = findfirst(x-> x == o, spp_retained)
        
        adj_new[sindx, oindx] = true
    end

    return SpeciesInteractionNetwork(Unipartite(spp_retained), Binary(adj_new))
end

gdf = groupby(df, [:foodweb_number, :sequence_number, :g])
pairs = DataFrame()
for g in gdf

    sort!(g, [:richness_pre], rev = true)

    if nrow(g) == 1 
        continue
    end

    pre  = sample_web(g[1, :alpha_web], g[1, :density_pre], 0.01)
    post = sample_web(g[2, :alpha_web], g[1, :density_pre], 0.01)

    density_pre = g[1, :density_pre]
    q10 = quantile(density_pre, 0.1)

    push!(pairs,
        (
            pre = pre, 
            post = post,
            extinction_proportion = g[1, :extinction_proportion],
            g = g[1, :g],
            density_pre = density_pre,
            q10 = q10
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

fig = Makie.Figure(size = (1200, 800)) 
ax  = Makie.Axis(fig[1,1]; 
    xlabel = "Cascade Size", 
    ylabel = "Proportion Shared Interactions"
)
empty!(ax)
Makie.scatter!(ax, pairs.q10, pairs.prop_shared)
Makie.boxplot!(ax, pairs.g, pairs.prop_shared, width = 0.01)