struct AlphaManifoldCallbackAffect

    groups::Vector{Vector{Int64}}

    function AlphaManifoldCallbackAffect(fwm::FoodwebModel)

        groups = get_index_groups(fwm)

        new(groups)
    end
end

# Return true -> Run after every timestep.
function condition!(u, t, integrator)

    return true
end

#Affect!
function (amca::AlphaManifoldCallbackAffect)(integrator)

    for group in amca.groups

        total = sum([integrator.u[x] for x in group])
        
        # Put each alpha back into [0, 1], and enforce sum(alphas) = 1
        for alpha in group

            integrator.u[alpha] /= total 
            u_modified!(integrator, true)
        end
    end
end

"""
For every consumer species, get a vector of the 
indices leading to all the alpha parameters.
"""
function get_index_groups(fwm::FoodwebModel)::Vector{Vector{Int64}}

    consumers = filter(x -> isconsumer(fwm, x), species(fwm))

    index_groups = []

    for sp in consumers

        # Get interactions from sp to their resources.
        is = filter(x -> !isloop(x) & (subject(x) == sp), interactions(fwm))
     
        # All alphas should be on all the edges in is, so we should get away
        # with looking at only the vars from the first interaction.
        dr = fwm.dynamic_rules[is[1]]

        alphas = filter(x -> is_trait_variable(fwm, x), dr.vars)

        indices = get_index.(Ref(fwm.vars), alphas)

        if isempty(indices)

            continue
        end

        push!(index_groups, indices)
    end

    return index_groups
end

function is_trait_variable(fwm, v)

    trait_variables = variables(fwm, type = TRAIT_VARIABLE)

    for t in trait_variables

        if isequal(t, v)

            return true
        end
    end

    return false
end

"""
    AlphaManifoldCallbackAffect

This is a callback to be used with the ODE Solver. It ensures that for each
consumer, the sum of all their alpha values remains one (or close to it). 

That is, the points are confined to a manifold defined by ∑α = 1.

This wouldn't of been needed in the original Kondoh model because of the use of
type I responses. The system would naturally maintain the constraint through
integration (maybe not though because of floating point error???). Brose's
addition of type II responses made this non-linear, which won't naturally
maintain the constraint.

There's probably a more "elegant" (mathematical rather than numerical) way of
fixing this, but I'm no mathematician so I have no idea how. This works.
"""
function AlphaManifoldCallback(fwm::FoodwebModel)

    amca = AlphaManifoldCallbackAffect(fwm)

    return DiscreteCallback(condition!, amca; save_positions = (false, true))
end