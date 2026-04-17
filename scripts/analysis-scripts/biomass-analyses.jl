include("../../src/HOI_Adaptive_Foraging.jl")
using .HOI_Adaptive_Foraging

using WGLMakie
using DataFrames
using CSV
using GLM
using Statistics
using SpeciesInteractionNetworks 
using HigherOrderFoodwebs

"""
By reducing over the path with * (and using Probabilistic networks) this isn't 
really a distance anymore? Maybe I should rename this function???
Oh well.
"""
function distancebetween(
    web::SpeciesInteractionNetwork{Unipartite{T}, Quantitative{Float64}}, 
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

function cascade_mean_density(web, cascade, u)::Float64

    if isempty(cascade)

        return 0
    end

    spp = last.(cascade)

    acc = 0
    for sp in spp

        acc += target_density(web, sp, u)
    end

    avg = mean(filter(!iszero, u))

    return (acc - avg) / avg 
end

function flux_variance(web)::Float64

    intxs = SpeciesInteractionNetworks.interactions(web)

    return var(abs.(last.(intxs)))
end

function total_flux(web)::Float64

    intxs = SpeciesInteractionNetworks.interactions(web)

    return sum(abs.(last.(intxs)))
end

function target_flux(web, target)::Float64

    intxs = filter(x-> target ∈ x,
        SpeciesInteractionNetworks.interactions(web)
    )

    return sum(abs.(last.(intxs)))
end

function nearby_flux(web, target, radius = 2)

    # prob_web = rescale_network(web)
    
    nhbs = neighbors(web, target, radius)
    dst = distancebetween.(Ref(web), Ref(target), nhbs)

    for i in eachindex(dst)

        if isinf(dst[i])
            dst[i] = 0
        end 
    end

    vals = [target_flux(web, sp) for sp in nhbs]

    return sum(abs.(vals .* dst))
end

function target_density(web, target, u)

    indx = findfirst(x->x==target, species(web))

    return u[indx]
end

function nearby_density(web, target, u, radius = 2)

    # prob_web = rescale_network(web)

    nhbs = neighbors(web, target, radius)
    dst = distancebetween.(Ref(web), Ref(target), nhbs)

    for i in eachindex(dst)

        if isinf(dst[i])
            dst[i] = 0
        end 
    end

    return sum(dst .* target_density.(Ref(web), nhbs, Ref(u)))
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

df = CSV.read("sim-output/niche-model-2026-03-16/data.csv", DataFrame)

preprocessing!(df)

filter!(:realized_web => !ismissing, df)
filter!(:density_pre => !ismissing, df)
filter!(:richness_pre => x-> x > 20, df)

transform!(df,
    [:density_pre, :target_species, :realized_web] =>
    ByRow((d, s, w) -> target_density(w, s, d) / sum(d)) =>
    :target_density
)

transform!(df,
    [:density_pre, :target_species, :realized_web] =>
    ByRow((d, s, w) -> nearby_density(w, s, d) / sum(d)) =>
    :nearby_density
)

transform!(df,
    [:target_species, :realized_web] =>
    ByRow((s, w) -> target_flux(w, s) / total_flux(w)) =>
    :target_flux
)

transform!(df,
    [:target_species, :realized_web] =>
    ByRow((s, w) -> nearby_flux(w, s) / total_flux(w)) =>
    :nearby_flux
)

transform!(df,
    [:realized_web] =>
    ByRow(flux_variance) =>
    :flux_variance
)

transform!(df,
    [:target_flux, :target_density] =>
    ByRow((f, d) -> log(f/d)) =>
    :target_turnover
)

# ------------------------------------------------------------------------------ #
# Target Density:                                                                #
# Is there a relationship between the density of the target species and the size #
# of the resultant cascade? Does g function differently in these cases?          #
# ------------------------------------------------------------------------------ #

filt = copy(df)
filter!(:extinction_proportion => !iszero, filt)

f  = Figure(size = (650, 650))
ax = Axis(f[1,1], xlabel = "Proportion Extinct", ylabel = "Target Density")

scatter!(ax, 
    filt[:, :extinction_proportion], 
    filt[:, :target_density]
)

save("figures/target_density_v_proportion_extinct.png", f)

f  = Figure(size = (650, 650))
ax = Axis(f[1,1], xlabel = "Proportion Extinct", ylabel = "Nearby Density")

scatter!(ax, 
    filt[:, :extinction_proportion], 
    filt[:, :nearby_density]
)

save("figures/target_density_v_proportion_extinct.png", f)

# ------------------------------------------------------------------------------ #
# Flux:                                                                
# ------------------------------------------------------------------------------ #

filt = copy(df)
filter!(:extinction_proportion => !iszero, filt)

transform!(filt,
    [:realized_web, :target_species] =>
    ByRow((web, sp) -> generality(web, sp)) =>
    :target_degree
)

f  = Figure(size = (650, 650))
ax = Axis(f[1,1], xlabel = "Proportion Extinct", ylabel = "Target Flux")

empty!(ax)
scatter!(ax, 
    filt[:, :extinction_proportion], 
    filt[:, :target_flux]
)

save("figures/target_flux_v_cascade_size.png", f)

f  = Figure(size = (650, 650))
ax = Axis(f[1,1], xlabel = "Proportion Extinct", ylabel = "Nearby Flux")

scatter!(ax, 
    filt[:, :extinction_proportion], 
    filt[:, :nearby_flux]
)

save("figures/nearby_flux_v_cascade_size.png", f)

# -------------------------------------------------- #
# Average density of the species which go extinct:   #
# Are the species that go extinct rare, or abundant? #
# -------------------------------------------------- #

transform!(df,
    [:realized_web, :cascade, :density_pre] =>
    ByRow((x,y,z) -> cascade_mean_density(x,y,z)) =>
    :cascade_mean_density
)

filt = copy(df)
filter!(:g => x-> x<0.25, filt)

f  = Figure(size = (1200, 650))
ax = Axis(f[1,1], 
    xlabel = "Proportion of community extinct", 
    ylabel = "mean density of secondarily extinct species",
    title  = "Low Adaptive Foraging (g < 0.25)"
)
scatter!(ax, 
    filt[:, :extinction_proportion],
    filt[:, :cascade_mean_density],
    # width = 0.05
)
xlims!(ax, [0.0, 0.4])

filt = copy(df)
filter!(:g => x-> x>0.25, filt)

ax = Axis(f[1,2], 
    xlabel = "Proportion of community extinct", 
    ylabel = "mean density of secondarily extinct species",
    title  = "High Adaptive Foraging (g > 0.25)"
)
scatter!(ax, 
    filt[:, :extinction_proportion],
    filt[:, :cascade_mean_density],
    # width = 0.05
)
xlims!(ax, [0.0, 0.4])

save("figures/mean_density_v_g.png", f)

#
# Is there more biomass in communities which have adaptive foraging.
# 
#

filt = copy(df)

transform!(filt,
    [:density_pre] =>
    ByRow(sum) =>
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
    width = 0.05
)

save("figures/biomass_v_g.png", f)

#
# Is biomass more even in communities with adaptive foraging?
# 
#

filt = copy(df)

transform!(filt,
    [:density_pre] =>
    ByRow(x -> quantile(filter(!iszero, x), 0.1)) =>
    :foodweb_biomass_q10
)

f  = Figure(size = (650, 650))
ax = Axis(f[1,1], 
    xlabel = "g", 
    ylabel = "10th Percentile of Species Biomass",
)
boxplot!(ax, 
    filt[:, :g],
    filt[:, :foodweb_biomass_q10],
    width = 0.05
)

ylims!(ax, [-0.0001, 0.005])

save("figures/biomass_q10_v_g.png", f)

#
#
#

