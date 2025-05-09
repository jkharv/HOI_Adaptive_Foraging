module HOI_Adaptive_Foraging

using HigherOrderFoodwebs
using SpeciesInteractionNetworks
using Distributions
using LinearAlgebra
using DataFrames

include("../src/modified_niche_model.jl")
export modified_niche_model

include("../src/measures.jl")
export median_interaction_strength, time_window_population_cv
export jacobian_interaction_strength, time_window_community_cv


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