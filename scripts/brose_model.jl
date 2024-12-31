function build_fwm(s, c, g)

    web = nichemodel(s, c)
    fwm = (FoodwebModel ∘ optimal_foraging)(web)

    # ---------------------------------------------------------------- #
    # Calculate some traits for each species to parametrize the model. #
    # ---------------------------------------------------------------- #

    traits = DataFrame(species = species(fwm))
    traits.vulnerability = vulnerability.(Ref(fwm.hg), traits.species);
    traits.generality = generality.(Ref(fwm.hg), traits.species);
    traits.trophic_level = distancetobase.(Ref(fwm.hg), traits.species, mean);

    # Body Mass
    f_mass(tl) = tl == 1.0 ? 1.0 : 100^(tl - 1);
    traits.mass = f_mass.(traits.trophic_level);

    # ------------------------------- #
    # Create species-level parameters #
    # ------------------------------- #

    traits.metabolic_rate    = add_param!.(Ref(fwm), :x, traits.species, 0.5);
    traits.growth_rate       = add_param!.(Ref(fwm), :r, traits.species, 1.0);
    traits.carrying_capacity = add_param!.(Ref(fwm), :k, traits.species, 1.0);
    traits.max_consumption   = add_param!.(Ref(fwm), :y, traits.species, 6.0);
    traits.adaptation_rate   = add_param!.(Ref(fwm), :g, traits.species, g);

    # --------------------------------------------------------- #
    # Subset the interactions for different parts of the model. #
    # --------------------------------------------------------- #

    growth = filter(isloop, interactions(fwm));
    producer_growth = filter(x -> isproducer(fwm, subject(x)), growth);
    consumer_growth = filter(x -> isconsumer(fwm, subject(x)), growth);
    trophic = filter(!isloop, interactions(fwm));

    # ----------------------------------------------------------------------- #
    # Set up a Dict to keep track of the attack rates for each interaction.   #
    # ----------------------------------------------------------------------- #

    atk_rates = Dict{Tuple{Symbol, Symbol}, Num}()

    for i ∈ trophic

        s = subject(i)
        o = object(i)

        sym = Symbol("α_$(s)_$(o)")
        var = add_var!(fwm, sym, fwm.t)

        atk_rates[(s, o)] = var 
    end

    # ----------------------------------------- #
    # Set up the dynamical aspects of the model #
    # ----------------------------------------- #

    for i ∈ producer_growth

        sbj = subject(i)
        s = fwm.conversion_dict[sbj]

        r = traits[traits.species .== sbj, :growth_rate][1]
        k = traits[traits.species .== sbj, :carrying_capacity][1]

        fwm.dynamic_rules[i] = DynamicRule(logistic(s,r,k))
    end

    for i ∈ consumer_growth

        sbj = subject(i)
        s = fwm.conversion_dict[sbj]

        x = traits[traits.species .== sbj, :metabolic_rate][1]

        fwm.dynamic_rules[i] = DynamicRule(-x * s)
    end

    F(Bj, B, ar, b0) = Bj / (b0 + sum(ar .* B))

    for i ∈ trophic
    
        s = fwm.conversion_dict[subject(i)]
        o = fwm.conversion_dict[object(i)]
        af_m = [fwm.conversion_dict[x] for x in with_role(:AF_modifier, i)]
        r = [o, af_m...]

        x = traits[traits.species .== subject(i), :metabolic_rate][1]
        y = traits[traits.species .== subject(i), :max_consumption][1]
        g = traits[traits.species .== subject(i), :adaptation_rate][1]
        
        a = atk_rates[(subject(i), object(i))]
        ar = [atk_rates[(subject(i), x)] for x in with_role(:AF_modifier, i)]
        ar = [a, ar...] 

        ar_norm = ar ./ sum(ar)
        a_norm = ar[1]

        object_gain = x * y * a_norm * F(o, r, ar_norm, 1.0)
        mean_gain = mean(x * y * ar_norm .* F.(r, Ref(r), Ref(ar_norm), Ref(1.0)))
        fwm.aux_dynamic_rules[a] = DynamicRule( 
            g * a_norm * (1 - a_norm) * (object_gain - mean_gain)
        )
        fwm.u0[a] = 1/length(r)

        fwd = a_norm * x * y * F(o, r, ar_norm, 1.0) * s
        bwd = -fwd

        fwm.dynamic_rules[i] = DynamicRule(fwd, bwd)
    end

    return assemble_foodweb(fwm, Rosenbrock23())
end
