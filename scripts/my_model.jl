"""
This function is still kinda shit cause it's asymptotes at the MAX value way past
x = 1.0 I need to get something that behaves a little better. I could just jack n 
way up but then the function isn't smooth enough around the middle.
"""
function scaled_assimilation_efficiency(niche_radius, niche_centre, trait_value)

    MAX = 0.8
    MIN = 0.05
    k   = 0.5
    n   = 4

    @assert abs(trait_value - niche_centre) <= niche_radius

    x = (niche_radius - abs(trait_value - niche_centre)) / niche_radius

    return (MAX - MIN) * (x^n / (k^n + x^n)) + MIN
end

function build_my_fwm(s, c, b, gval)

    web, _, traits = HOI_Adaptive_Foraging.niche_model_min_basal(s, c, b)
    fwm = (FoodwebModel ∘ optimal_foraging)(web)

    # ---------------------------------------------------------------- #
    # Calculate some traits for each species to parametrize the model. #
    # ---------------------------------------------------------------- #

    traits.vulnerability = vulnerability.(Ref(fwm.hg), traits.species);
    traits.generality = generality.(Ref(fwm.hg), traits.species);
    traits.trophic_level = distancetobase.(Ref(fwm.hg), traits.species, mean);

    # Niche stuff
    traits.niche_diameter = traits.niche_upper .- traits.niche_lower;
    traits.niche_radius = traits.niche_diameter ./ 2
    traits.niche_centre = traits.niche_lower .+ traits.niche_radius

    # Body Mass
    mass_ratio = 1000;
    f_mass(tl) = mass_ratio^tl;
    traits.mass = f_mass.(traits.trophic_level);

    # ------------------------------- #
    # Just one global adaptation_rate #
    # ------------------------------- #

    g = add_param!(fwm , :g, Vector{Symbol}(), gval)

    # ------------------------------- #
    # Create species-level parameters #
    # ------------------------------- #

    traits.metabolic_rate    = add_param!.(Ref(fwm), :x, traits.species,
        0.314 * traits.mass.^(-0.25)
    );
    traits.growth_rate       = add_param!.(Ref(fwm), :r, traits.species, 1.0);
    traits.carrying_capacity = add_param!.(Ref(fwm), :k, traits.species, 1.0);
    traits.max_consumption   = add_param!.(Ref(fwm), :y, traits.species, 4.0);

    # --------------------------------------------------------- #
    # Subset the interactions for different parts of the model. #
    # --------------------------------------------------------- #

    growth = filter(isloop, interactions(fwm));
    producer_growth = filter(x -> isproducer(fwm, subject(x)), growth);
    consumer_growth = filter(x -> isconsumer(fwm, subject(x)), growth);
    trophic = filter(!isloop, interactions(fwm));

    # --------------------------------------------------------- #
    #  Set up some eij parameters for each trophic interaction  #
    # --------------------------------------------------------- #

    assimilation_efficiencies = Dict{Interaction, Num}();

    for intx ∈ trophic

        sym = Symbol("$(subject(intx))_$(object(intx))")
    
        s = subject(intx)
        o = object(intx)

        s_niche_centre = traits[traits.species .== s, :niche_centre][1]
        s_niche_radius  = traits[traits.species .== s, :niche_radius][1]
        o_trait_value  = traits[traits.species .== o, :trait_value][1]

        e_value = scaled_assimilation_efficiency(
            s_niche_radius, 
            s_niche_centre, 
            o_trait_value
        )

        p = add_param!(fwm, :e, sym, e_value)

        assimilation_efficiencies[intx] = p
    end

    # --------------------------------------------------- #
    # Add equations to keep track of dynamic attack rates #
    # --------------------------------------------------- #

    attack_rates = Dict{Tuple{Symbol, Symbol}, Num}()

    for i ∈ trophic

        s = subject(i)
        o = object(i)

        sym = Symbol("a_$(s)_$(o)")
        var = add_var!(fwm, sym, TRAIT_VARIABLE)

        attack_rates[(s, o)] = var 
    end

    # ----------------------------------------- #
    # Set up the dynamical aspects of the model #
    # ----------------------------------------- #

    for i ∈ producer_growth

        sbj = subject(i)
        s = get_variable(fwm, sbj)

        r = traits[traits.species .== sbj, :growth_rate][1]
        k = traits[traits.species .== sbj, :carrying_capacity][1]

        fwm.dynamic_rules[i] = DynamicRule(
            s*r*(1 - s/k)
        )
    end

    for i ∈ consumer_growth

        sbj = subject(i)
        s = get_variable(fwm, sbj)

        x = traits[traits.species .== sbj, :metabolic_rate][1]

        fwm.dynamic_rules[i] = DynamicRule(
            -x * s
        )
    end

    F(B_focal, B_resources, a_resources, b0) = (B_focal * a_resources[1]) / (b0 + a_resources ⋅ B_resources);

    for i ∈ trophic

        
        s = get_variable(fwm, subject(i))
        o = get_variable(fwm, object(i))
        m = [get_variable(fwm, x) for x in with_role(:AF_modifier, i)]
        r = [o, m...]

        x = traits[traits.species .== subject(i), :metabolic_rate][1]
        y = traits[traits.species .== subject(i), :max_consumption][1]

        e = assimilation_efficiencies[i]

        a = attack_rates[(subject(i), object(i))]
        ar = [attack_rates[(subject(i), x)] for x in with_role(:AF_modifier, i)]
        ar = [a, ar...] 

        ar_norm = ar ./ sum(ar)

        object_gain = x * y * F(o, r, ar_norm, 0.5)
        mean_gain = mean(x * y * F.(r, Ref(r), Ref(ar_norm), Ref(0.5)))

        fwm.aux_dynamic_rules[a] = DynamicRule( 
            g * a * (1 - a) * (object_gain - mean_gain)
        )

        set_u0!(fwm, Dict(a => 1/length(r)))

        fwd = x * y * s * F(o, r, ar_norm, 0.5) * e
        bwd = -1.0 * fwd / e 

        fwm.dynamic_rules[i] = DynamicRule(fwd, bwd)
    end

    return (traits, fwm, assemble_foodweb(fwm; extra_transient_time = 2000))
end