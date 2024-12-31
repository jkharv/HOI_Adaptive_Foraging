function stable_fwm_factory()

    web, adj, traits = modified_niche_model(30, 0.15)

    traits.vulnerability = vulnerability.(Ref(web), traits.species);
    traits.generality = generality.(Ref(web), traits.species);
    traits.trophic_level = distancetobase.(Ref(web), traits.species, mean);

    f_mass(tl) = tl == 1.0 ? 1.0 : 100^(tl -1)
    traits.mass = f_mass.(traits.trophic_level)

    # IDK why someone whould have div zeros produces Infs. They should clearly be NaN.
    traits.ω = 1.0 ./ traits.generality
    traits.ω = map(x-> isinf(x) ? NaN : x, traits.ω)

    # # Apply a NTP structural model for the multi-species functional response and 
    # # adaptive foraging.
    web = optimal_foraging(web)
    fwm = FoodwebModel(web)

    growth = filter(isloop, interactions(fwm))

    isprod(x) = isproducer(fwm, x)
    iscons(x) = isconsumer(fwm, x)

    producer_growth = filter(isprod ∘ subject, growth)
    consumer_growth = filter(iscons ∘ subject, growth)

    trophic = filter(!isloop, interactions(fwm))

    for i ∈ producer_growth
        
        sbj = with_role(:subject, i)

        s = @group fwm i sbj begin

            x -> r * x * (1- x/k) - m * x
        end r ~ 1.0 m ~ 0.01 k ~ 1.0

        @set_rule fwm i s 
    end

    for i ∈ consumer_growth

        sbj = with_role(:subject, i)
        sbj_mass = traits[traits.species .== sbj, :mass][1]

        s = @group fwm i sbj begin
            
            x -> -m * x 
        end m ~ 0.01 * sbj_mass

        @set_rule fwm i s
    end

    for i ∈ trophic

        sbj = with_role(:subject, i)
        sbj_mass = traits[traits.species .== sbj, :mass][1]
        sbj_tl = traits[traits.species .== sbj, :trophic_level][1]

        # Is it a herbivore?
        e_ij = sbj_tl == 2.0 ? 0.45 : 0.85
        obj = with_role(:object, i)
        mods = with_role(:AF_modifier, i)
        w = traits[traits.species .== sbj, :ω][1]

        s = @group fwm i sbj begin

            x -> m * y * x

        end m ~ (0.01 * sbj_mass) y ~ 8.0

        o = @group fwm i [obj..., mods...] begin

            x[] -> (ω * x[1])/(b0 + sum(ω .* x[2:end]))
            x[] -> -(ω * x[1])/(b0 + sum(ω .* x[2:end])) / e

        end ω ~ w b0 ~ 0.5 e ~ e_ij

        @set_rule fwm i s * o
    end

    u0 = Dict(species(fwm) .=> rand(Uniform(0.5, 1.0), length(species(fwm))))
    set_initial_condition!(fwm, u0)

    extinction_cb = ExtinctionThresholdCallback(1e-20)
    steady_state_cb = TerminateSteadyState(1e-5, 1e-2)
    cb = CallbackSet(extinction_cb, steady_state_cb)

    sol = solve(fwm, RK4(); callback = cb, tspan = (0, 1000));

    new_u0 = Dict(species(fwm) .=> sol[end])

    new_fwm = FoodwebModel(
        fwm.hg, 
        fwm.dynamic_rules,
        fwm.t,
        fwm.vars,
        Dict{Num, Number}(),
        fwm.params,
        fwm.param_vals,
        missing,
        missing
    )

    set_initial_condition!(new_fwm, new_u0)

    return new_fwm
end