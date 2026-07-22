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
using DataStructures
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
    densities = densities ./ sum(u)

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

df1 = CSV.read("sim-output/1-extinctions-2026-07-07/data.csv", DataFrame)
df4 = CSV.read("sim-output/4-extinctions-2026-07-07/data.csv", DataFrame)

df = vcat(df1, df4)

preprocessing!(df)

filter!(:realized_web => !ismissing, df)
filter!(:density_pre => !ismissing, df)
filter!(:richness_pre => x-> x > 20, df)

transform!(df,
    [:realized_web, :realized_trim, :realized_prob, :target_species, :density_pre] =>
    ByRow((qweb, tweb, pweb, t, d) -> target_density(qweb, tweb, pweb, t, d) / sum(d)) =>
    :target_density
)

# transform!(df,
#     [:realized_web, :realized_trim, :realized_prob, :target_species, :density_pre] =>
#     ByRow((qweb, tweb, pweb, t, d) -> nearby_density(qweb, tweb, pweb, t, d) / sum(d)) =>
#     :nearby_density,
#     threads = true
# )

# transform!(df,
#     [:realized_web, :realized_trim, :realized_prob, :target_species] =>
#     ByRow(
#         (qweb, tweb, pweb, t) -> 
#             target_flux(qweb, tweb, pweb, t) / total_flux(qweb, tweb, pweb)
#     ) => :target_flux
# )

# transform!(df,
#     [:realized_web, :realized_trim, :realized_prob, :target_species] =>
#     ByRow(
#         (qweb, tweb, pweb, t) -> 
#             nearby_flux(qweb, tweb, pweb, t) / total_flux(qweb, tweb, pweb)
#         ) =>
#     :nearby_flux
# )

# transform!(df,
#     [:target_flux, :target_density] =>
#     ByRow((f, d) -> f/d) =>
#     :target_turnover
# )

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

transform!(df,
    [:realized_web, :realized_trim, :realized_prob, :cascade, :density_pre] =>
    ByRow(
        (qweb, tweb, pweb, c, d) -> 
            cascade_density(qweb, tweb, pweb, x->quantile(x, 0.9), c, d)
        ) =>
    :cascade_q90_density
)

transform!(df,
    [:realized_web, :realized_trim, :realized_prob, :cascade, :density_pre] =>
    ByRow(
        (qweb, tweb, pweb, c, d) -> 
            cascade_density(qweb, tweb, pweb, maximum, c, d)
        ) =>
    :cascade_max_density
)

# --------------------------------------------------------- #
# Are large cascades dominated by rare or abundant species. #
# --------------------------------------------------------- #

filt = copy(df)
filter!(:extinction_proportion => !iszero, filt)

f  = Figure(size = (1200, 800))
ax = Axis(f[1,1], 
    xlabel = "Proportion extinct", 
    ylabel = "Median density of secondarily extinct species",
)

scatter!(ax, 
    filt[:, :extinction_proportion],
    filt[:, :cascade_median_density],
)

# ----------------------------------------------------------- #
# Is biomass more even in communities with adaptive foraging? #
# ----------------------------------------------------------- # 

ax = Axis(f[1,2], 
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
    gdf = combine(gdf, :density => x -> quantile(x, 0.1))

    x = lines!(ax, gdf[:, :g], gdf[:, :density_function], linewidth = 3)
    push!(ls, x)
    push!(ns, n)
end

Legend(f[1,3], ls, string.(ns), "Primary Extinctions")

save("figures/biomass-eveness-g.png", f)

# ------------------------------------------------------------------------ #
# Is there more total biomass in communities which have adaptive foraging? #
# A little bit, yes.                                                       #
# ------------------------------------------------------------------------ #

filt = copy(df)
filter!(:richness_pre => x-> x> 20, filt)

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

# ----------------------------------------------------------------- #
# Do the Rank-Abundance curves differ between large/small cascades? #
# ----------------------------------------------------------------- #

function rank_abundance(df)
    
    acc = [zeros(Float64, 0) for i in 1:length(df.density_pre[1])]
    target_ranks = Vector{Int64}()

    for row in eachrow(df)

        densities = sort(row.density_pre, rev = true)
        densities = OrderedDict(
            species(df.realized_web[1]) .=> df.density_pre[1]
        )
        sort!(densities, byvalue = true, rev = true)

        for (i, (spp, d)) in enumerate(densities)

            push!(acc[i], d)
        end
    end

    avg   = mean.(acc)
    stdev = std.(acc)
    ns    = length.(acc)

    stderror = stdev ./ sqrt.(ns)

    return (
        lower = log.(avg .- stderror),
        avg   = log.(avg),
        upper = log.(avg + stderror),
        target_ranks
    )

end

function extinction_ranks(df)

    acc = zeros(Float64, length(df.density_pre[1]))

    for row in eachrow(df)

        densities = sort(row.density_pre, rev = true)
        densities = OrderedDict(
            species(df.realized_web[1]) .=> df.density_pre[1]
        )
        sort!(densities, byvalue = true, rev = true)

        ranks = Dict(keys(densities) .=> 1:length(df.density_pre[1]))

        extinctions = last.(row.cascade)

        for sp in extinctions 
            
            sp = sp[1]

            acc[ranks[sp]] += 1
        end

    end

    return acc / sum(acc)
end

fig = Figure(size = (800, 800))
ax  = Axis(fig[1,1], xlabel = "Rank", ylabel = "Log Density")

# Small cascades
filt = copy(df)
filter!(:richness_pre => x-> x >= 20, filt)
filter!(:extinction_proportion => x-> x < 0.2, filt)

curve = rank_abundance(filt)
ranks = extinction_ranks(filt)
band!(ax, collect(1:30), curve.lower, curve.upper)
small = lines!(ax, collect(1:30), curve.avg, color = ranks, 
    linewidth = 10, linestyle = :dash
)

# Large cascades
filt = copy(df)
filter!(:richness_pre => x-> x >= 20, filt)
filter!(:extinction_proportion => x-> x >= 0.2, filt)

curve = rank_abundance(filt)
ranks = extinction_ranks(filt)
band!(ax, collect(1:30), curve.lower, curve.upper)
large = lines!(ax, collect(1:30), curve.avg, color = ranks, linewidth = 10)

Legend(fig[1,2], 
    [small, large], 
    ["Small", "Large"], 
    "Cascade Size"
)

save("figures/rank-abundance-cascade-size.png", fig)

# ---------------------------------------------------------- #
# Do the Rank-Abundance curves differ between large/small g? #
# ---------------------------------------------------------- #

fig = Figure(size = (800, 800))
ax  = Axis(fig[1,1], xlabel = "Rank", ylabel = "Log Density")

# No g 
filt = copy(df)
filter!(:richness_pre => x-> x >= 20, filt)
filter!(:g => x-> x == 0, filt)

curve = rank_abundance(filt)
ranks = extinction_ranks(filt)
band!(ax, collect(1:30), curve.lower, curve.upper, color = ranks)
nog = lines!(ax, collect(1:30), curve.avg, color = ranks, linewidth = 10)

# Large g
filt = copy(df)
filter!(:richness_pre => x-> x >= 20, filt)
filter!(:g => x-> x > 0.2, filt)

curve = rank_abundance(filt)
ranks = extinction_ranks(filt)
band!(ax, collect(1:30), curve.lower, curve.upper)
yesg = lines!(ax, collect(1:30), curve.avg, color = ranks, linewidth = 10, linestyle = :dot)

Legend(fig[1,2], 
    [nog, yesg], 
    ["No AF", "Strong AF"], 
    "Adaptive Foraging"
)

save("figures/rank-abundance-g.png", fig)