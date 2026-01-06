module HOI_Adaptive_Foraging

using HigherOrderFoodwebs
using SpeciesInteractionNetworks
using AnnotatedHypergraphs
using Distributions
using LinearAlgebra
using DataFrames
using DiffEqCallbacks
using SparseArrays
using SciMLBase

include("../src/niche_model_min_basal.jl")
export niche_model_min_basal

include("../src/rectangular_web.jl")
export rectangular_web

include("../src/measures.jl")
export median_interaction_strength, time_window_population_cv
export eigenstability, time_window_community_cv, richness

include("../src/alpha_manifold.jl")
export AlphaManifoldCallback

end