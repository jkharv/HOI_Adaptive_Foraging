include("src/HOI_Adaptive_Foraging.jl")

using .HOI_Adaptive_Foraging



c = treatments(
    g = 0:0.01:0.5,
    c = 0.01:0.02:0.4,
    e = 0.01:0.02:0.4
)

simulations(
    species_richness = 10,
    minimum_basal_species = 1,
    number_of_foodwebs = 5,
    number_of_sequences = 2,
    n_extinctions = 2,
    stem = "test",
)