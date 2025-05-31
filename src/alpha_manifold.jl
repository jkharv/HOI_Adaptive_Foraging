function g(fwm, attack_rates, consumers, resid, u, p)

    for (i, sbj) in enumerate(consumers)

        sum = 0 

        for obj in species(fwm)

            if haskey(attack_rates, (sbj, obj))

                alpha = attack_rates[(sbj, obj)]
                index = get_index(fwm.vars, alpha)
                sum += u[index]
            end
        end

        resid[i] = 1 - sum
    end
end

function AlphaManifold(fwm, alphas)

    consumers = filter(x -> isconsumer(fwm, x), species(fwm))

    # The residual vector is the size of the number of consumers
    residual_prototype = zeros(Float64, length(consumers))

    _g(resid, u, p) = g(fwm, alphas, consumers, resid, u, p)

    return ManifoldProjection(_g;
        autodiff = AutoForwardDiff(),
        resid_prototype = residual_prototype
    )
end
