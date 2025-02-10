module HOI_Adaptive_Foraging

using ModelingToolkit
using HigherOrderFoodwebs
using SpeciesInteractionNetworks
using Distributions
using DifferentialEquations
using DiffEqCallbacks
using Random
using LinearAlgebra
using DataFrames
using SymbolicIndexingInterface
using WGLMakie

include("../src/modified_niche_model.jl")
export modified_niche_model


include("../src/foodweb_timeseries_plot.jl")
export foodweb_timeseries

# include("../src/pairwise_network_sample.jl")
# export pairwise_network_sample

include("../src/extinction_threshold_callback.jl")
include("../src/richness_termination_callback.jl")
export extinctionthresholdcallback2, RichnessTerminationCallback



# Count the number of basal species in the foodweb.
function count_basal(web::SpeciesInteractionNetwork)

    count(iszero, (values ∘ generality)(web))
end

# Niche model with a minimum guaranteed number of basal species
function niche_model_min_basal(s, c, b)

    for i ∈ 1:100

        web = HOI_Adaptive_Foraging.modified_niche_model(s, c)

        if count_basal(first(web)) >= b

            return web
        end
    end

    throw(ErrorException("Couldn't find a web with $b basal species."))
end

export niche_model_min_basal

end
