include("../../src/HOI_Adaptive_Foraging.jl")
using .HOI_Adaptive_Foraging

using WGLMakie
using DataFrames
using CSV
using GLM
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

function cascade_mean_density(
    qweb::SpeciesInteractionNetwork{Unipartite{T}, Quantitative{Float64}},
    tweb::SpeciesInteractionNetwork{Unipartite{T}, Quantitative{Float64}},
    pweb::SpeciesInteractionNetwork{Unipartite{T}, Probabilistic{Float64}},

    cascade::Vector{Tuple{Float64, Vector{T}}}, 
    u::Vector{Float64}
    )::Float64 where T

    if isempty(cascade)

        return 0
    end

    spp = last.(cascade)

    acc = 0
    for sp in spp

        acc += target_density(qweb, tweb, pweb, sp, u)
    end

    avg = mean(filter(!iszero, u))

    return (acc - avg) / avg 
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

df1 = CSV.read("sim-output/new-tl-algorithm-2026-04-20/data.csv", DataFrame)
df2 = CSV.read("sim-output/two-extinctions-2026-04-21/data.csv", DataFrame)

df = vcat(df1, df2)

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
    :nearby_density
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
            cascade_mean_density(qweb, tweb, pweb, c, d)
        ) =>
    :cascade_mean_density
)

# ------------------------------------------------------------------------------ #
# Target Density:                                                                #
# Is there a relationship between the density of the target species and the size #
# of the resultant cascade? Does g function differently in these cases?          #
# ------------------------------------------------------------------------------ #

function gridify(xs, ys, nx, ny, xmax, ymax)

    w = xmax / nx
    h = ymax / ny

    m = zeros(Int64, nx, ny)

    xs = Integer.(floor.((xs ./ w) .+ 1))
    ys = Integer.(floor.((ys ./ h) .+ 1))

    for (x, y) in zip(xs, ys)

        m[y,x] += 1
    end

    return m
end

filt = copy(df)
filter!(:extinction_proportion => !iszero, filt)
filter!(:extinction_proportion => x-> x>0.2, filt)
filter!(:n_targets => x -> x == 1, filt)

f  = Figure(size = (1200, 750))
ax = Axis(f[1,1], xlabel = "Proportion Extinct", ylabel = "Target Density")

# scatter!(ax,
#     filt[:, :extinction_proportion],
#     filt[:, :target_density]
# )

n = 12
m = gridify(filt[:, :extinction_proportion], filt[:, :target_density], n, n, 0.5, 1.0)
mp = m ./ sum(m)
mp = log.(mp)
heatmap!(ax, 0.2:(0.5/10):0.5, 0:(1.0/10):1.0, mp)

ax = Axis(f[1,2], xlabel = "Proportion Extinct", ylabel = "Nearby Density")

# scatter!(ax,
#     filt[:, :extinction_proportion],
#     filt[:, :nearby_density]
# )

m = gridify(filt[:, :extinction_proportion], filt[:, :nearby_density], n, n, 0.5, 1.0)
mp = m ./ sum(m)
mp = log.(mp)
heatmap!(ax, 0:(0.5/10):0.5, 0:(1.0/10):1.0, mp)

# Two targets

filt = copy(df)
filter!(:extinction_proportion => !iszero, filt)
filter!(:n_targets => x -> x == 2, filt)
ax = Axis(f[2,1], xlabel = "Proportion Extinct", ylabel = "Target Density")

# scatter!(ax, 
#     filt[:, :extinction_proportion], 
#     filt[:, :target_density]
# )

m = gridify(filt[:, :extinction_proportion], filt[:, :target_density], n, n, 0.5, 1.0)
mp = m ./ sum(m)
mp = log.(mp)
heatmap!(ax, 0:(0.5/10):0.5, 0:(1.0/10):1.0, mp)

ax = Axis(f[2,2], xlabel = "Proportion Extinct", ylabel = "Nearby Density")

m = gridify(filt[:, :extinction_proportion], filt[:, :nearby_density], n, n, 0.5, 1.0)
mp = m ./ sum(m)
mp = log.(mp)
heatmap!(ax, 0:(0.5/10):0.5, 0:(1.0/10):1.0, mp)

# scatter!(ax,
#     filt[:, :extinction_proportion], 
#     filt[:, :nearby_density]
# )

save("figures/density_v_proportion_extinct_heatmap.png", f)

# ------------------------------------------------------------------------------ #
# Flux:                                                                
# ------------------------------------------------------------------------------ #

filt = copy(df)
filter!(:extinction_proportion => !iszero, filt)
filter!(:n_targets => x -> x == 1, filt)

f  = Figure(size = (1000, 650))
ax = Axis(f[1,1],  ylabel = "Target Flux")

# scatter!(ax, 
#     filt[:, :extinction_proportion], 
#     filt[:, :target_flux]
# )

m = gridify(filt[:, :extinction_proportion], filt[:, :target_flux], n, n, 0.5, 1.0)
mp = m ./ sum(m)
mp = log.(mp)
heatmap!(ax, 0:(0.5/10):0.5, 0:(1.0/10):1.0, mp)

ax = Axis(f[1,2], ylabel = "Nearby Flux")

# scatter!(ax, 
#     filt[:, :extinction_proportion], 
#     filt[:, :nearby_flux]
# )

m = gridify(filt[:, :extinction_proportion], filt[:, :nearby_flux], n, n, 0.5, 2.0)
mp = m ./ sum(m)
mp = log.(mp)
heatmap!(ax, 0:(0.5/10):0.5, 0:(1.0/10):2.0, mp)

# now with two targets

filt = copy(df)
filter!(:extinction_proportion => !iszero, filt)
filter!(:n_targets => x -> x == 2, filt)

ax = Axis(f[2,1], xlabel = "Proportion Extinct", ylabel = "Target Flux")

# scatter!(ax, 
#     filt[:, :extinction_proportion], 
#     filt[:, :target_flux]
# )

m = gridify(filt[:, :extinction_proportion], filt[:, :target_flux], n, n, 0.5, 1.0)
mp = m ./ sum(m)
mp = log.(mp)
heatmap!(ax, 0:(0.5/10):0.5, 0:(1.0/10):1.0, mp)

ax = Axis(f[2,2], xlabel = "Proportion Extinct", ylabel = "Nearby Flux")

# scatter!(ax, 
#     filt[:, :extinction_proportion], 
#     filt[:, :nearby_flux]
# )

m = gridify(filt[:, :extinction_proportion], filt[:, :nearby_flux], n, n, 0.5, 2.0)
mp = m ./ sum(m)
mp = log.(mp)
heatmap!(ax, 0:(0.5/10):0.5, 0:(1.0/10):2.0, mp)

save("figures/flux_v_cascade_size_heatman.png", f)

# -------------------------------------------------- #
# Average density of the species which go extinct:   #
# Are the species that go extinct rare, or abundant? #
# -------------------------------------------------- #

filt = copy(df)
filter!(:n_targets => x -> x == 1, filt)
filter!(:g => x-> x<0.25, filt)
filter!(:extinction_proportion => !iszero, filt)

f  = Figure(size = (1000, 650))
ax = Axis(f[1,1], 
    xlabel = "Proportion of community extinct", 
    ylabel = "mean density of secondarily extinct species",
    title  = "Low Adaptive Foraging (g < 0.25)"
)

m = gridify(filt[:, :extinction_proportion], filt[:, :nearby_flux], n, n, 0.5, 25)
mp = m ./ sum(m)
mp = log.(mp)
heatmap!(ax, 0:(0.5/10):0.5, 0:(1.0/10):25, mp)

scatter!(ax, 
    filt[:, :extinction_proportion],
    filt[:, :cascade_mean_density],
    # width = 0.05
)
xlims!(ax, [0.0, 0.5])

filt = copy(df)
filter!(:n_targets => x -> x == 2, filt)
filter!(:g => x-> x>0.25, filt)
filter!(:extinction_proportion => !iszero, filt)

ax1 = Axis(f[1,2], 
    xlabel = "Proportion of community extinct", 
    ylabel = "mean density of secondarily extinct species",
    title  = "High Adaptive Foraging (g > 0.25)"
)
scatter!(ax1, 
    filt[:, :extinction_proportion],
    filt[:, :cascade_mean_density],
    # width = 0.05
)
xlims!(ax1, [0.0, 0.5])

# Now with two targets

filt = copy(df)
filter!(:n_targets => x -> x == 1, filt)
filter!(:g => x-> x<0.25, filt)
filter!(:extinction_proportion => !iszero, filt)

ax = Axis(f[2,1], 
    xlabel = "Proportion of community extinct", 
    ylabel = "mean density of secondarily extinct species",
    title  = "Low Adaptive Foraging (g < 0.25)"
)
scatter!(ax, 
    filt[:, :extinction_proportion],
    filt[:, :cascade_mean_density],
    # width = 0.05
)
xlims!(ax, [0.0, 0.5])

filt = copy(df)
filter!(:n_targets => x -> x == 2, filt)
filter!(:g => x-> x>0.25, filt)
filter!(:extinction_proportion => !iszero, filt)

ax1 = Axis(f[2,2], 
    xlabel = "Proportion of community extinct", 
    ylabel = "mean density of secondarily extinct species",
    title  = "High Adaptive Foraging (g > 0.25)"
)
scatter!(ax1, 
    filt[:, :extinction_proportion],
    filt[:, :cascade_mean_density],
    # width = 0.05
)
xlims!(ax1, [0.0, 0.5])

save("figures/mean_density_v_g.png", f)

# ------------------------------------------------------------------ #
# Is there more biomass in communities which have adaptive foraging? #
# ------------------------------------------------------------------ #

filt = copy(df)

transform!(filt,
    [:density_pre] =>
    ByRow(log ∘ sum) =>
    :foodweb_total_biomass
)

f  = Figure(size = (650, 650))
ax = Axis(f[1,1], 
    xlabel = "g", 
    ylabel = "Food Web Total Biomass",
)
boxplot!(ax, 
    filt[:, :g],
    filt[:, :foodweb_total_biomass],
    width = 0.05,
    dodge = filt[:, :n_targets],
    color = map(d-> d == 1 ? :blue : :red, filt[:, :n_targets]),
)

save("figures/biomass_v_g.png", f)

# ----------------------------------------------------------- #
# Is biomass more even in communities with adaptive foraging? #
# ----------------------------------------------------------- # 

filt = copy(df)

transform!(filt,
    [:density_pre] =>
    ByRow(x -> log(quantile(filter(!iszero, x), 0.1))) =>
    :foodweb_biomass_q10
)

f  = Figure(size = (650, 650))
ax = Axis(f[1,1], 
    xlabel = "g", 
    ylabel = "Log 10th Percentile of Species Biomass",
)
boxplot!(ax, 
    filt[:, :g],
    filt[:, :foodweb_biomass_q10];
    width = 0.05,
    dodge = filt[:, :n_targets],
    color = map(d-> d == 1 ? :blue : :red, filt[:, :n_targets]),
)
ylims!(ax, [-25, 0])

save("figures/biomass_log_q10_v_g.png", f)
