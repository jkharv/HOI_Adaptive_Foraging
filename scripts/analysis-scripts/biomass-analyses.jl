include("../../src/HOI_Adaptive_Foraging.jl")
using .HOI_Adaptive_Foraging

using WGLMakie
using DataFrames
using CSV
using GLM
using StatsBase
using Distributions
using Statistics
using SpeciesInteractionNetworks 
using HigherOrderFoodwebs

function distance_weight(
    web::SpeciesInteractionNetwork{Unipartite{T}, Probabilistic{Float64}}, 
    sp1::T, 
    sp2::T
    )::Float64 where T

    forward = Inf
    reverse = Inf

    try

        forward = pathbetween(web, sp1, sp2)
        forward = reduce(*, last.(forward))
    catch

        forward = Inf
    end

    try

        reverse = pathbetween(web, sp2, sp1)
        reverse = reduce(*, last.(reverse))
    catch

        reverse = Inf
    end

    return min(forward, reverse)
end

function cascade_density(
    qweb::SpeciesInteractionNetwork{Unipartite{T}, Quantitative{Float64}},
    tweb::SpeciesInteractionNetwork{Unipartite{T}, Quantitative{Float64}},
    pweb::SpeciesInteractionNetwork{Unipartite{T}, Probabilistic{Float64}},
    summary_func::Function,
    cascade::Vector{Tuple{Float64, Vector{T}}}, 
    u::Vector{Float64}
    )::Float64 where T

    if isempty(cascade)

        return 0
    end

    spp = last.(cascade)

    densities = [target_density(qweb, tweb, pweb, sp, u) for sp in spp]
    densities = densities ./ mean(densities)

    return summary_func(densities)
end

function total_flux(
    qweb::SpeciesInteractionNetwork{Unipartite{T}, Quantitative{Float64}},
    tweb::SpeciesInteractionNetwork{Unipartite{T}, Quantitative{Float64}},
    pweb::SpeciesInteractionNetwork{Unipartite{T}, Probabilistic{Float64}},
)::Float64 where T

    intxs = SpeciesInteractionNetworks.interactions(qweb)

    return sum(abs.(last.(intxs)))
end

function target_flux(
    qweb::SpeciesInteractionNetwork{Unipartite{T}, Quantitative{Float64}},
    tweb::SpeciesInteractionNetwork{Unipartite{T}, Quantitative{Float64}},
    pweb::SpeciesInteractionNetwork{Unipartite{T}, Probabilistic{Float64}},
    targets::Vector{T}
    )::Float64 where T

    intxs = Set{eltype(SpeciesInteractionNetworks.interactions(qweb))}()

    for target in targets

        zs = filter(x-> target ∈ x,
            SpeciesInteractionNetworks.interactions(qweb)
        )

        [push!(intxs, z) for z in zs]
    end

    intxs = collect(intxs)

    return sum(abs.(last.(intxs)))
end

function nearby_flux(
    qweb::SpeciesInteractionNetwork{Unipartite{T}, Quantitative{Float64}}, 
    tweb::SpeciesInteractionNetwork{Unipartite{T}, Quantitative{Float64}}, 
    pweb::SpeciesInteractionNetwork{Unipartite{T}, Probabilistic{Float64}},
    targets::Vector{T}, 
    radius = 2
    ) where T

    nhbs = neighbors.(Ref(pweb), targets, Ref(radius))
    nhbs = (collect ∘ union)(nhbs...)

    dst = zeros(Float64, length(nhbs))

    # Distance from each neighbhour to the nearest target.
    for (i, n) in enumerate(nhbs)
        
        dst[i] = minimum(distance_weight.(Ref(pweb), targets, Ref(n)))
    end

    for i in eachindex(dst)

        if isinf(dst[i])
            dst[i] = 0
        end 
    end

    vals = [
            target_flux(qweb, tweb, pweb, [sp]) 
            for sp in nhbs
        ]

    return sum(abs.(vals .* dst))
end

function target_density(
    qweb::SpeciesInteractionNetwork{Unipartite{T}, Quantitative{Float64}}, 
    tweb::SpeciesInteractionNetwork{Unipartite{T}, Quantitative{Float64}}, 
    pweb::SpeciesInteractionNetwork{Unipartite{T}, Probabilistic{Float64}},
    targets::Vector{T}, 
    u::Vector{Float64})::Float64 where T

    indx = [findfirst(x->x==target, species(qweb)) for target in targets]

    return sum([u[i] for i in indx])
end

function nearby_density(
    qweb::SpeciesInteractionNetwork{Unipartite{T}, Quantitative{Float64}}, 
    tweb::SpeciesInteractionNetwork{Unipartite{T}, Quantitative{Float64}}, 
    pweb::SpeciesInteractionNetwork{Unipartite{T}, Probabilistic{Float64}},
    targets::Vector{T}, 
    u::Vector{Float64}, 
    radius = 2
    )::Float64 where T

    nhbs = Vector{Set{T}}(undef, length(targets))
    for (i, t) in enumerate(targets)

        nhbs[i] = neighbors(tweb, t, radius)
    end
    nhbs = (collect ∘ union)(nhbs...)

    dst = zeros(Float64, length(nhbs))

    # Distance from each neighbhour to the nearest target.
    for (i, n) in enumerate(nhbs)
        
        dst[i] = minimum(distance_weight.(Ref(pweb), targets, Ref(n)))
    end

    for i in eachindex(dst)

        if isinf(dst[i])
            dst[i] = 0
        end 
    end

    densities = zeros(Float64, length(nhbs))

    for i in eachindex(densities)

        densities[i] = target_density(qweb, tweb, pweb, [nhbs[i]], u)

    end

    return sum(dst .* densities)
end

function SpeciesInteractionNetworks.neighbors(web, target, r)

    if r == 1

        return neighbors(web, target)
    end

    n = neighbors(web, target)

    if isempty(n)
        return Set{Symbol}()
    end

    return union(neighbors.(Ref(web), n, Ref(r-1))...)
end

#
#
#

df1 = CSV.read("sim-output/one-extinction-2026-04-30/data.csv", DataFrame)
df2 = CSV.read("sim-output/two-extinctions-2026-04-30/data.csv", DataFrame)
df3 = CSV.read("sim-output/three-extinctions-2026-04-30/data.csv", DataFrame)
df4 = CSV.read("sim-output/four-extinctions-2026-04-30/data.csv", DataFrame)

df = vcat(df1, df2, df3, df4)

preprocessing!(df)

filter!(:realized_web => !ismissing, df)
filter!(:density_pre => !ismissing, df)
filter!(:richness_pre => x-> x > 20, df)

transform!(df,
    [:realized_web, :realized_trim, :realized_prob, :target_species, :density_pre] =>
    ByRow((qweb, tweb, pweb, t, d) -> target_density(qweb, tweb, pweb, t, d) / sum(d)) =>
    :target_density
)

transform!(df,
    [:realized_web, :realized_trim, :realized_prob, :target_species, :density_pre] =>
    ByRow((qweb, tweb, pweb, t, d) -> nearby_density(qweb, tweb, pweb, t, d) / sum(d)) =>
    :nearby_density,
    threads = true
)

transform!(df,
    [:realized_web, :realized_trim, :realized_prob, :target_species] =>
    ByRow(
        (qweb, tweb, pweb, t) -> 
            target_flux(qweb, tweb, pweb, t) / total_flux(qweb, tweb, pweb)
    ) => :target_flux
)

transform!(df,
    [:realized_web, :realized_trim, :realized_prob, :target_species] =>
    ByRow(
        (qweb, tweb, pweb, t) -> 
            nearby_flux(qweb, tweb, pweb, t) / total_flux(qweb, tweb, pweb)
        ) =>
    :nearby_flux
)

transform!(df,
    [:target_flux, :target_density] =>
    ByRow((f, d) -> f/d) =>
    :target_turnover
)

transform!(df,
    [:realized_web, :realized_trim, :realized_prob, :cascade, :density_pre] =>
    ByRow(
        (qweb, tweb, pweb, c, d) -> 
            cascade_density(qweb, tweb, pweb, mean, c, d)
        ) =>
    :cascade_mean_density
)

transform!(df,
    [:realized_web, :realized_trim, :realized_prob, :cascade, :density_pre] =>
    ByRow(
        (qweb, tweb, pweb, c, d) -> 
            cascade_density(qweb, tweb, pweb, median, c, d)
        ) =>
    :cascade_median_density
)

# ------------------------------------------------------------------------------ #
# Target Density:                                                                #
# Is there a relationship between the density of the target species and the size #
# of the resultant cascade? Does g function differently in these cases?          #
# ------------------------------------------------------------------------------ #

filt = copy(df)
filter!(:extinction_proportion => !iszero, filt)
filter!(:n_targets => x -> x == 1, filt)

f  = Figure(size = (1200, 750))
ax = Axis(f[1,1], xlabel = "Target Density", ylabel = "Proportion Extinct")

scatter!(ax,
    filt[:, :target_density],
    filt[:, :extinction_proportion]
)

ax = Axis(f[1,2], xlabel = "Nearby Density", ylabel = "Proportion Extinct")

scatter!(ax,
    filt[:, :nearby_density],
    filt[:, :extinction_proportion]
)

save("figures/density_v_proportion_extinct_one_target.png", f)

# Two targets

filt = copy(df)
filter!(:extinction_proportion => !iszero, filt)
filter!(:n_targets => x -> x == 2, filt)

f  = Figure(size = (1200, 750))
ax = Axis(f[1,1], xlabel = "Proportion Extinct", ylabel = "Target Density")

scatter!(ax, 
    filt[:, :extinction_proportion], 
    filt[:, :target_density]
)

ax = Axis(f[1,2], xlabel = "Proportion Extinct", ylabel = "Nearby Density")

scatter!(ax,
    filt[:, :extinction_proportion], 
    filt[:, :nearby_density]
)

save("figures/density_v_proportion_extinct_two_targets.png", f)

# Three targets

filt = copy(df)
filter!(:extinction_proportion => !iszero, filt)
filter!(:n_targets => x -> x == 3, filt)

f  = Figure(size = (1200, 750))
ax = Axis(f[1,1], xlabel = "Proportion Extinct", ylabel = "Target Density")

scatter!(ax, 
    filt[:, :extinction_proportion], 
    filt[:, :target_density]
)

ax = Axis(f[1,2], xlabel = "Proportion Extinct", ylabel = "Nearby Density")

scatter!(ax,
    filt[:, :extinction_proportion], 
    filt[:, :nearby_density]
)

save("figures/density_v_proportion_extinct_three_targets.png", f)

# Three targets

filt = copy(df)
filter!(:extinction_proportion => !iszero, filt)
filter!(:n_targets => x -> x == 4, filt)

f  = Figure(size = (1200, 750))
ax = Axis(f[1,1], xlabel = "Proportion Extinct", ylabel = "Target Density")

scatter!(ax, 
    filt[:, :extinction_proportion], 
    filt[:, :target_density]
)

ax = Axis(f[1,2], xlabel = "Proportion Extinct", ylabel = "Nearby Density")

scatter!(ax,
    filt[:, :extinction_proportion], 
    filt[:, :nearby_density]
)

save("figures/density_v_proportion_extinct_four_targets.png", f)

# ------------------------------------------------------------------------------ #
# Target Indegree:                                                                #
# Is there a relationship between the density of the target species and the size #
# of the resultant cascade? Does g function differently in these cases?          #
# ------------------------------------------------------------------------------ #

filt = copy(df)
filter!(:extinction_proportion => !iszero, filt)
filter!(:n_targets => x -> x == 1, filt)

f  = Figure(size = (1200, 750))
ax = Axis(f[1,1], xlabel = "Proportion Extinct", ylabel = "Target In-degree")

scatter!(ax,
    filt[:, :extinction_proportion],
    filt[:, :avg_target_indegree]
)

ax = Axis(f[1,2], xlabel = "Proportion Extinct", ylabel = "Target Out-degree")

scatter!(ax,
    filt[:, :extinction_proportion],
    filt[:, :avg_target_outdegree]
)

# ------------------------------------------------------------------------------ #
# Flux:                                                                
# ------------------------------------------------------------------------------ #

filt = copy(df)
filter!(:extinction_proportion => !iszero, filt)
filter!(:n_targets => x -> x == 1, filt)

f  = Figure(size = (1000, 650))
ax = Axis(f[1,1],  
    ylabel = "Extinction Proportion",
    xlabel = "Target Flux"
)

scatter!(ax, 
    filt[:, :target_flux], 
    filt[:, :extinction_proportion]
)

ax = Axis(f[1,2], xlabel = "Nearby Flux")

scatter!(ax, 
    filt[:, :nearby_flux], 
    filt[:, :extinction_proportion]
)

# now with two targets

filt = copy(df)
filter!(:extinction_proportion => !iszero, filt)
filter!(:n_targets => x -> x == 2, filt)

ax = Axis(f[2,1], xlabel = "Proportion Extinct", ylabel = "Target Flux")

scatter!(ax, 
    filt[:, :extinction_proportion], 
    filt[:, :target_flux]
)

ax = Axis(f[2,2], xlabel = "Proportion Extinct", ylabel = "Nearby Flux")

scatter!(ax, 
    filt[:, :extinction_proportion], 
    filt[:, :nearby_flux]
)

save("figures/flux_v_cascade_size.png", f)

# --------------------------------------------------------- #
# Are large cascades dominated by rare or abundant species. #
# --------------------------------------------------------- #

filt = copy(df)
# filter!(:n_targets => x -> x == 1, filt)
filter!(:extinction_proportion => !iszero, filt)

f  = Figure(size = (1000, 650))
ax = Axis(f[1,1], 
    xlabel = "Proportion extinct", 
    ylabel = "Median density of secondarily extinct species",
)

scatter!(ax, 
    filt[:, :extinction_proportion],
    filt[:, :cascade_median_density],
)
xlims!(ax, [2/20, 0.7])

save("figures/median-cascade-densisty-v-cascade-size.png", f)

# ------------------------------------------------------------------ #
# Is there more biomass in communities which have adaptive foraging? #
# ------------------------------------------------------------------ #

filt = copy(df)
filter!(:n_targets => x -> x == 2, filt)

transform!(filt,
    [:density_pre] =>
    ByRow(log ∘ sum) =>
    :foodweb_total_biomass
)

f  = Figure(size = (1000, 650))
ax = Axis(f[1,1], 
    xlabel = "Strength of Adaptive Foraging", 
    ylabel = "Log Food Web Total Biomass",
)
boxplot!(ax, 
    filt[:, :g],
    filt[:, :foodweb_total_biomass],
    width = 0.05 
)

fit = lm(@formula(foodweb_total_biomass ~ g), filt)
ablines!(ax, coef(fit)[1], coef(fit)[2], color = :red, linewidth = 4)

save("figures/biomass_v_g.png", f)

# ----------------------------------------------------------- #
# Is biomass more even in communities with adaptive foraging? #
# ----------------------------------------------------------- # 

f  = Figure(size = (1200, 750))
ax = Axis(f[1,1], 
    xlabel = "Adaptation Rate", 
    ylabel = "10th Percentile of Species Density",
    xticks = unique(filt[:, :g]),
    xtickformat = "{:.2f}"
)

ls = []
ns = []
for n in unique(df[:, :n_targets])

    filt = copy(df)
    filter!(:n_targets => x -> x == n, filt)

    biomasses = DataFrame()
    for row in eachrow(filt)

        for d in row[:density_pre]

            if !iszero(d)
                push!(biomasses, (g = row[:g], density = d))
            end
        end
    end

    gdf = groupby(biomasses, :g)
    gdf = combine(gdf, :density => x->quantile(x, 0.1))

    x = lines!(ax, gdf[:, :g], gdf[:, :density_function], linewidth = 3)
    push!(ls, x)
    push!(ns, n)
end

Legend(f[1,2], ls, string.(ns), "Primary Extinctions")

save(".figures/biomass-eveness-g.png", f)