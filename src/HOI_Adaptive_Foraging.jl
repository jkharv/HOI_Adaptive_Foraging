module HOI_Adaptive_Foraging

using HigherOrderFoodwebs
using SpeciesInteractionNetworks
using Distributions
using LinearAlgebra
using DataFrames
using DiffEqCallbacks
using SparseArrays
using SciMLBase

include("../src/modified_niche_model.jl")
export modified_niche_model, niche_model_min_basal

include("../src/measures.jl")
export median_interaction_strength, time_window_population_cv
export eigenstability, time_window_community_cv, richness

include("../src/alpha_manifold.jl")
export AlphaManifoldCallback

end