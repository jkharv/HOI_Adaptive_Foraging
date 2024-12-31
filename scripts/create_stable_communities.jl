using Revise
using HOI_Adaptive_Foraging
using HigherOrderFoodwebs
using JLD2

n = 100
fw_names = ["Foodweb $n" for n ∈ 1:n]
fws = Dict{String, HigherOrderFoodwebs.FoodwebModel{Symbol}}()

Threads.@threads for i ∈ fw_names

    fws[i] = stable_fwm_factory()
end

save("stable_foodwebs.jld2", fws)