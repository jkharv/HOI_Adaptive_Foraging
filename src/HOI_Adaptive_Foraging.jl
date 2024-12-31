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
using CairoMakie

include("../src/modified_niche_model.jl")
export modified_niche_model


include("../src/foodweb_timeseries_plot.jl")
export foodweb_timeseries


include("../src/extinction_threshold_callback.jl")
include("../src/richness_termination_callback.jl")
export extinctionthresholdcallback2, RichnessTerminationCallback

end