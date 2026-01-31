module HOI_Adaptive_Foraging

using SpeciesInteractionNetworks
using AnnotatedHypergraphs
using HigherOrderFoodwebs

using Distributions
using LinearAlgebra
using DataFrames

using DiffEqCallbacks
using OrdinaryDiffEqTsit5

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

using SparseArrays
using Dates

include("../src/niche_model_min_basal.jl")
export niche_model_min_basal

include("../src/rectangular_web.jl")
export rectangular_web

include("../src/allometric_fwm.jl")
export build_fwm

include("../src/measures.jl")
export median_interaction_strength, time_window_population_cv
export eigenstability, time_window_community_cv, richness
export cascade_timespan, eigencentrality_of_spp
export cascade_trophic_range, secondary_extinctions_during_trial
export mean_extinction_time

include("../src/simulation_tools.jl")
export make_output_dir!, save_parameters!, simulation_batch, extinction_indices

include("../src/alpha_manifold.jl")
export AlphaManifoldCallback

end