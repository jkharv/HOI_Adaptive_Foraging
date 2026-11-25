include("../../src/HOI_Adaptive_Foraging.jl")
using .HOI_Adaptive_Foraging

using WGLMakie
using DataFrames
using CSV
using JLD2
using GLM
using StatsBase
using Distributions
using Statistics
using SpeciesInteractionNetworks 
using DataStructures
using HigherOrderFoodwebs

function cascade_density(
    qweb::SpeciesInteractionNetwork{Unipartite{T}, Quantitative{Float64}},
    tweb::SpeciesInteractionNetwork{Unipartite{T}, Quantitative{Float64}},
    pweb::SpeciesInteractionNetwork{Unipartite{T}, Probabilistic{Float64}},
    cascade::Vector{Tuple{Float64, Vector{T}}}, 
    u::Vector{Float64}
    )::Tuple{Float64, Float64, Float64} where T

    if isempty(cascade)

        return (0.0, 0.0, 0.0)
    end

    spp = last.(cascade)

    densities = [target_density(qweb, tweb, pweb, sp, u) for sp in spp]
    densities = densities ./ sum(u)

    null_samples = Vector{Float64}(undef, 10_000) 
    Threads.@threads for i in (collect ∘ eachindex)(null_samples)

        nz_spp = findall(!iszero, u)
        idxs = rand(nz_spp, length(spp))
        null_densities = u[idxs]
        null_densities = null_densities / sum(u)
        null_samples[i] = median(null_densities)
    end

    standard_error = std(null_samples; corrected = false)

    return  (median(densities), median(null_samples), standard_error)
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

df1 = CSV.read("sim-output/one-extinction-2026-09-09.csv", DataFrame)
jld1 = jldopen("sim-output/one-extinction-2026-09-09.jld2")
# df4 = CSV.read("sim-output/long-4-extinctions-2026-07-30.csv", DataFrame)
# jld4 = jldopen("sim-output/long-4-extinctions-2026-07-30.jld2")

df_disturbances = CSV.read("sim-output/disturbances-2026-09-02.csv", DataFrame)
jld_disturbances = jldopen("sim-output/disturbances-2026-09-02.jld2")

df_no_disturbances = CSV.read("sim-output/no-disturbances-2026-09-02.csv", DataFrame)
jld_no_disturbances = jldopen("sim-output/no-disturbances-2026-09-02.jld2")

preprocessing!(df1, jld1)
preprocessing!(df4, jld4)

preprocessing!(df_disturbances, jld_disturbances)
preprocessing!(df_no_disturbances, jld_no_disturbances)

df_disturbances.removal_fraction .= 0.9
df_no_disturbances.removal_fraction .= 0.0

df = vcat(df1, df4)
df_dist = vcat(df_disturbances, df_no_disturbances)

filter!(:realized_web => !ismissing, df)
filter!(:density_pre => !ismissing, df)

filter!(:realized_web => !ismissing, df_dist)
filter!(:density_pre => !ismissing, df_dist)

transform!(df,
    [:realized_web, :realized_trim, :realized_prob, :target_species, :density_pre] =>
    ByRow((qweb, tweb, pweb, t, d) -> target_density(qweb, tweb, pweb, t, d) / sum(d)) =>
    :target_density
)

transform!(df,
    [:realized_web, :realized_trim, :realized_prob, :cascade, :density_pre] =>
    ByRow(
        (qweb, tweb, pweb, c, d) -> 
            cascade_density(qweb, tweb, pweb, c, d)
        ) =>
    [:cascade_density, :cascade_density_null, :cascade_density_se]
)

transform!(df,
    [:cascade_density, :cascade_density_null, :cascade_density_se] =>
    ByRow((x, null, se) -> (x - null) / se)
    => :cascade_density_standardized  
)

# ---------------------------------------------- #
# Are larger cascades dominated by rare species? #
# Yes                                            #
# ---------------------------------------------- #

filt = copy(df)
filter!(:cascade_number => x-> x == 1, filt)
filter!(:secondary_extinctions => x-> x>0, filt)

f  = Figure(size = (800, 800))
ax = Axis(f[1,1], 
    xlabel = "Cascade Size", 
    ylabel = "Median Cascade Density (SE)",
    xtickformat = "{:.2f}"
)

scatter!(ax, 
    filt[:, :extinction_proportion], 
    filt[:, :cascade_density_standardized],
)

#This probly needs to be a WLS regression.
fit = lm(@formula(cascade_density_standardized ~ extinction_proportion), filt)
ablines!(ax, coef(fit)[1], coef(fit)[2], color = :red, linewidth = 4)

save("figures/cascaade_density_v_cascade_size.png", f)

# -------------------------------------------------------------------------- #
# Does the rarity of the species in a cascade change with adaptive foraging? #
# Yes, they get a little less rare.                                          #
# -------------------------------------------------------------------------- #

filt = copy(df)
filter!(:cascade_number => x-> x == 1, filt)
filter!(:secondary_extinctions => x-> x>0, filt)

f  = Figure(size = (800, 800))
ax = Axis(f[1,1], 
    xlabel = "Adaptation Rate", 
    ylabel = "Median Cascade Density (SE)",
    xtickformat = "{:.2f}"
)

boxplot!(ax, 
    filt[:, :g], 
    filt[:, :cascade_density_standardized],
    width = 0.05
)

save("figures/cascade_density_v_g.png", f)

# ---------------------------------------------------------------------- #
# Are rare species less rare in communities that have adaptive foraging? #
# Yes                                                                    #
# ---------------------------------------------------------------------- # 

filt = copy(df)
filter!(:cascade_number => x-> x==1, filt)

f  = Figure(size = (800, 800))
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
    gdf = combine(gdf, :density => (x -> quantile(x, 0.1)) => :density_function)
    # gdf = combine(gdf, :density => shannon_eveness => :density_function)

    x = lines!(ax, gdf[:, :g], gdf[:, :density_function], linewidth = 3)
    push!(ls, x)
    push!(ns, n)
end

Legend(f[1,2], ls, string.(ns), "Primary Extinctions")

save("figures/biomass-eveness-g.png", f)

# ------------------------------------------------------------------------ #
# Is there more total biomass in communities which have adaptive foraging? #
# A little bit, yes.                                                       #
# ------------------------------------------------------------------------ #

filt = copy(df)
filter!(:cascade_number => x-> x==5, filt)

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

# interesting thing is the trend revereses depending on whether I only use the
# first cascade or all the cascades.

# -------------------------------------------------------------------- #
# Are communities with adaptive foraging more even than those without? #
# Yes, a little bit.                                                   #
# -------------------------------------------------------------------- #

filt = copy(df)
filter!(:cascade_number => x-> x==1, filt)

transform!(filt,
    [:density_pre] =>
    ByRow(shannon_eveness) =>
    :eveness
)

f  = Figure(size = (1000, 650))
ax = Axis(f[1,1], 
    xlabel = "Strength of Adaptive Foraging", 
    ylabel = "Shannon Eveness",
)
boxplot!(ax, 
    filt[:, :g],
    filt[:, :eveness],
    width = 0.05 
)

fit = lm(@formula(eveness ~ g), filt)
ablines!(ax, coef(fit)[1], coef(fit)[2], color = :red, linewidth = 4)

save("figures/eveness_v_g.png", f)

# ----------------------------------------------------- #
# Do more even communities experience smaller cascades? #
# ----------------------------------------------------- #

filt = copy(df)
filter!(:cascade_number => x-> x==1, filt)

transform!(filt,
    [:density_pre] =>
    ByRow(shannon_eveness) =>
    :eveness
)

f  = Figure(size = (1000, 650))
ax = Axis(f[1,1], 
    xlabel = "Cascade Size", 
    ylabel = "Shannon Eveness",
)
scatter!(ax, 
    filt[:, :extinction_proportion],
    filt[:, :eveness],
)

fit = lm(@formula(eveness ~ extinction_proportion), filt)
ablines!(ax, coef(fit)[1], coef(fit)[2], color = :red, linewidth = 4)

save("figures/eveness_v_cascade_size.png", f)

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
filter!(:cascade_number => x-> x < 3, filt)
filter!(:extinction_proportion => x-> x < 0.2, filt)

curve = rank_abundance(filt)
small = lines!(ax, collect(1:30), curve.avg, 
    linewidth = 10, linestyle = :dash
)

# Large cascades
filt = copy(df)
filter!(:cascade_number => x-> x < 3, filt)
filter!(:extinction_proportion => x-> x >= 0.2, filt)

curve = rank_abundance(filt)
large = lines!(ax, collect(1:30), curve.avg, linewidth = 10)

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
filter!(:cascade_number => x-> x == 1, filt)
filter!(:g => x-> x == 0, filt)

curve = rank_abundance(filt)
ranks = extinction_ranks(filt)
band!(ax, collect(1:30), curve.lower, curve.upper, color = ranks)
nog = lines!(ax, collect(1:30), curve.avg, color = ranks, linewidth = 10)

# Large g
filt = copy(df)
filter!(:cascade_number => x-> x == 1, filt)
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

#
# Does the rank abundance curve change over time
#

fig = Figure(size = (800, 800))
ax  = Axis(fig[1,1], xlabel = "Rank", ylabel = "Log Density")

ncascade = [1,2,3,4]
ls = [] 
for n in ncascade

    filt = copy(df)
    filter!(:cascade_number => x-> x == n, filt)
    filter!(:g => x-> x == 0, filt)

    curve = rank_abundance(filt)
    ranks = extinction_ranks(filt)
    band!(ax, collect(1:30), curve.lower, curve.upper)
    l = lines!(ax, collect(1:30), curve.avg, linewidth = 10)
    push!(ls, l)
end

Legend(fig[1,2], 
    ls, 
    string.(ncascade), 
    "Cascade Number"
)