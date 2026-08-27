# affect! function
function affect!(
    integrator, 
    n_spp::Int64, 
    removal_fraction::Float64,
    disturbances::Vector{Tuple{Float64, Vector{T}}} 
    ) where T

    @assert (removal_fraction >= 0.0) & (removal_fraction <= 1.0)

    fwm = integrator.sol.prob.f.sys

    idxs = findall(!iszero, integrator.u[1:richness(fwm)])

    if length(idxs) < n_spp
        n_spp = length(idxs)
    end

    target_species = rand(idxs, n_spp)

    for t in target_species

        integrator.u[t] -= integrator.u[t] * removal_fraction
    end

    u_modified!(integrator, true)

    spp_symbols = [species(fwm)[i] for i in target_species]

    push!(disturbances, (integrator.t, spp_symbols))

    return
end

function disturbance_condition(disturbance_times, u, t, integrator)

    return t ∈ disturbance_times
end

function DisturbanceCallback(
    fwm::FoodwebModel{T},
    n_spp::Int64,
    disturbance_times::Vector{Float64};
    removal_fraction = 0.5,
    disturbance_history = Vector{Tuple{Float64, Vector{T}}}()
    ) where T

    @assert n_spp <= richness(fwm)

    c(u, t, integrator) = disturbance_condition(disturbance_times, u, t, integrator)
    a(integrator) = affect!(integrator, n_spp, removal_fraction, disturbance_history)

    return DiscreteCallback(c, a)
end