module HOI_Adaptive_Foraging

const EXTINCTION_INTERVAL = 2000.0
const EXTINCTION_THRESHOLD = 1e-10
export EXTINCTION_INTERVAL, EXTINCTION_THRESHOLD

using SpeciesInteractionNetworks
using AnnotatedHypergraphs
using HigherOrderFoodwebs

using Distributions
using DataFrames
using JLD2
using Dates

using DiffEqCallbacks
using OrdinaryDiffEqTsit5
using FiniteDiff
using LinearAlgebra
using SparseArrays

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

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
export mean_extinction_time, maximum_trophic_level

include("../src/simulation_tools.jl")
export make_output_dir!, save_parameters!
export simulation_batch, extinction_indices, treatments

include("../src/alpha_manifold.jl")
export AlphaManifoldCallback

include("../src/preprocessing.jl")
export preprocessing!

end