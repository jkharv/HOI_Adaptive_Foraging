using SpeciesInteractionNetworks
using HigherOrderFoodwebs
using DataFrames
using CSV
using GLM
using Statistics
# using WGLMakie
using LinearAlgebra
using MultivariateStats
using JLD2

function HigherOrderFoodwebs.isproducer(web::SpeciesInteractionNetwork, sp)::Bool

    @assert sp ∈ species(web)

    for intx ∈ interactions(web)

        if isloop(intx)

            continue
        elseif subject(intx) == sp

            return false
        end
    end

    return true
end

function count_basal_species(web::SpeciesInteractionNetwork)

    count = 0

    for sp in species(web)

        if isproducer(web, sp)
            count += 1
        end
    end

    return count
end

function network_out_spectrum(web::SpeciesInteractionNetwork)

    eigenvals = (eigen ∘ outlaplacian_matrix)(web).values
    sort!(eigenvals, by = (abs ∘ real), rev = true)

    return eigenvals
end

function network_in_spectrum(web::SpeciesInteractionNetwork)

    eigenvals = (eigen ∘ inlaplacian_matrix)(web).values
    sort!(eigenvals, by = (abs ∘ real), rev = true)

    return eigenvals
end

function adjacency_matrix(web::SpeciesInteractionNetwork)

    m = zeros(Bool, (richness(web), richness(web)))

    for intx in interactions(web)

        i = findfirst(x -> x == subject(intx), species(web))
        j = findfirst(x -> x == object(intx), species(web))
        
        m[i,j] = true

        if i == j
            m[i,j] = false
        end
    end

    return m
end

function indegree_matrix(web::SpeciesInteractionNetwork)

    m = zeros(Int64, (richness(web), richness(web)))
    
    for sp in species(web)

        i = findfirst(x -> x == sp, species(web))
        m[i,i] = count(x -> subject(x) == sp, interactions(web))
    end

    return m
end

function outdegree_matrix(web::SpeciesInteractionNetwork)

    m = zeros(Int64, (richness(web), richness(web)))
    
    for sp in species(web)

        i = findfirst(x -> x == sp, species(web))
        m[i,i] = count(x -> object(x) == sp, interactions(web))
    end

    return m
end

function inlaplacian_matrix(web::SpeciesInteractionNetwork)

    return indegree_matrix(web) - adjacency_matrix(web)
end

function outlaplacian_matrix(web::SpeciesInteractionNetwork)

    return outdegree_matrix(web) - adjacency_matrix(web)
end

#=
I should do a simulation run with less species, 30 or so, and a larger number of
foodwebs and sequences in each foodweb. Potentially less values of g as well.
The idea is to maximize the power to see a signal of foodweb structure on
secondary extinctions and the effect of adaptive foraging. There needs to be a 
bit of a trade-off since assembling new foodwebs is a computationally intensive 
task.
=#

DATA_DIR = "simulation-output-2025-09-24"
df = CSV.read(joinpath(DATA_DIR, "data.csv"), DataFrame)

filter!(:richness_pre => x-> x >= 5, df)
filter!(:retcode => x -> x == "Success", df)
# Add a column for proportion of community gone extinct after a primary extinction.
transform!(df, [:richness_pre, :secondary_extinctions] => 
    ((x, y) -> y ./ x) => 
    :extinction_proportion
)
transform!(df, [:foodweb_number, :sequence_number] => 
    ((x,y) -> string.(x) .* "-" .* string.(y)) =>
    :fw_x_seq
)

dircontent = readdir(DATA_DIR)
foodweb_files = filter(x -> contains(x, r"foodweb*"), dircontent)

webs = []
traits = []

for file in foodweb_files

    jldopen(joinpath(DATA_DIR, file)) do f
   
        push!(webs, f["web"])    
        push!(traits, f["traits"])
    end
end

basal_species = []
for web in webs

    b = count_basal_species(web)
    push!(basal_species, b)
end

gdf = groupby(df, :foodweb_number);
mean_extinctions = combine(gdf, :secondary_extinctions => mean);
median_extinction_proportion = combine(gdf, :extinction_proportion => median)
mean_extinction_proportion = combine(gdf, :extinction_proportion => mean)
max_extinction_proportion = combine(gdf, :extinction_proportion => maximum)

leftjoin!(mean_extinctions, median_extinction_proportion; on = :foodweb_number)
leftjoin!(mean_extinctions, mean_extinction_proportion; on = :foodweb_number)
leftjoin!(mean_extinctions, max_extinction_proportion; on = :foodweb_number)

mean_extinctions[:, :basal_species] = basal_species

spectrums = [network_in_spectrum(x) for x in webs]
spectrums = reduce(hcat, spectrums)

pca_fit = fit(PCA, real.(spectrums))
pca_coords = predict(pca_fit, real.(spectrums))

using WGLMakie

t_spec = (transpose ∘ real)(spectrums)
t_pca  = transpose(pca_coords)

fig = Figure(size = (600, 600))
ax = Axis(fig[1,1],
    xlabel = "PC1",
    ylabel = "PC2"
)
empty!(ax)

#  "foodweb_number"
#  "secondary_extinctions_mean"
#  "extinction_proportion_median"
#  "extinction_proportion_mean"
#  "extinction_proportion_maximum"

scatter!(ax, t_spec[:, 1:2];
    color = mean_extinctions[:, :extinction_proportion_maximum], 
    colormap = :heat,
    markersize  = 20.0
); 

empty!(ax)
scatter!(ax, t_pca[:, 1:2];
    color = mean_extinctions[:, :extinction_proportion_maximum],
    colormap = :heat,
    markersize  = 20.0
)