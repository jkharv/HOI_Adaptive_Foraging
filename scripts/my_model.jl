"""
This function is still kinda shit cause it's asymptotes at the MAX value way past
x = 1.0 I need to get something that behaves a little better. I could just jack n 
way up but then the function isn't smooth enough around the middle.
"""
function scaled_assimilation_efficiency(niche_radius, niche_centre, trait_value)

    MAX = 0.8
    MIN = 0.2
    k   = 0.5
    n   = 4

    @assert abs(trait_value - niche_centre) <= niche_radius

    x = (niche_radius - abs(trait_value - niche_centre)) / niche_radius

    return (MAX - MIN) * (x^n / (k^n + x^n)) + MIN
end

function holling_disk(B_focal, a_focal, B_all, a_all, b0)

    return (B_focal * a_focal) / (b0 + (a_all ⋅ B_all))
end

function build_my_fwm(s, c, b, gval)

    web, _, traits = niche_model_min_basal(s, c, b)
    fwm = (FoodwebModel ∘ optimal_foraging)(web)

    # --------------- #
    # Add some params #
    # --------------- #

    traits.vulnerability = vulnerability.(Ref(fwm.hg), traits.species);
    traits.generality = generality.(Ref(fwm.hg), traits.species);
    traits.trophic_level = distancetobase.(Ref(fwm.hg), traits.species, mean);

    # Niche stuff
    traits.niche_diameter = traits.niche_upper .- traits.niche_lower;
    traits.niche_radius = traits.niche_diameter ./ 2
    traits.niche_centre = traits.niche_lower .+ traits.niche_radius

    # Body Mass
    mass_ratio = 1000;
    traits.mass = map(x -> mass_ratio^x, traits.trophic_level)

    # ---------------------------------------------------------------------------- #
    # Just one global adaptation rate here. Kondoh and Brose already played aroun
    # with different proportions of adaptive foraging and there was nothing super
    # interesting so there's not point redoing all that here as well.
    # ---------------------------------------------------------------------------- #

    g = add_param!(fwm , :g, gval)

    # ------------------------------- #
    # Create species-level parameters #
    # ------------------------------- #

    traits.metabolic_rate = 
        add_param!.(Ref(fwm), :x, traits.species, 0.314 * traits.mass.^(-0.25));
    traits.growth_rate       = add_param!.(Ref(fwm), :r, traits.species, 1.0);
    traits.carrying_capacity = add_param!.(Ref(fwm), :k, traits.species, 1.0);
    traits.max_consumption   = add_param!.(Ref(fwm), :y, traits.species, 4.0);

    # ------------------- #
    # Subset interactions #
    # ------------------- #

    growth = filter(isloop, interactions(fwm))
    trophic = filter(!isloop, interactions(fwm))
    pgrowth = filter(x-> isproducer(fwm, subject(x)), growth)
    cgrowth = filter(x-> isconsumer(fwm, subject(x)), growth)

    # --------------------------------------- #
    # Add trait vars for foraging preferences #
    # --------------------------------------- #
    attack_rates = Dict{AnnotatedHyperedge, Symbol}()

    for intx in trophic

        s = subject(intx)
        o = object(intx)
        m = with_role(:AF_modifier, intx)

        if isempty(m)

            init_val = 1.0
        else

            init_val = 1.0 / (length(m)+1)
        end

        # Unambiguous symbol.
        sym = Symbol(:alpha,s,o)
        var = add_var!(fwm.vars, sym, TRAIT_VARIABLE, init_val)

        attack_rates[intx] = var
    end

    reverse_atk = Dict(map(reverse, collect(attack_rates)))
    atk_groups = Dict{Symbol, Vector{Symbol}}()

    for sp in species(fwm) 

        sp_trophic_intrxs = filter(x -> subject(x) == sp, trophic)

        x = [attack_rates[x] for x in sp_trophic_intrxs]

        for a in x
            
            atk_groups[a] = [a, setdiff(x, [a])...]
        end
    end

    # ------------------------------------------------------- #
    # Add assimilation efficiencies scaled to niche position. #
    # ------------------------------------------------------- #

    assimilation_efficiencies = Dict{Interaction, Symbol}();

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

    # -------------- #
    # Add Some Rules #
    # -------------- #

    @rule fwm for intx in pgrowth
    
        @var s = subject(intx)
        @param r = traits[traits.species .== subject(intx), :growth_rate][1]
        @param k = traits[traits.species .== subject(intx), :carrying_capacity][1]

        return s * r * (1 - s / k)
    end

    @rule fwm for intx in cgrowth

        @var s = subject(intx)
        @param x = traits[traits.species .== subject(intx), :metabolic_rate][1]

        return -s*x 
    end

    @rule fwm for intx in trophic

        @var s = subject(intx)
        @var o = object(intx)
        @var m = with_role(:AF_modifier, intx)

        @var a = attack_rates[intx]
        @var ar = atk_groups[attack_rates[intx]]

        @param x = traits[traits.species .== subject(intx), :metabolic_rate][1]
        @param y = traits[traits.species .== subject(intx), :max_consumption][1]
        @param e = assimilation_efficiencies[intx]

        ar_norm = ar ./ Ref(sum(ar))

        fr = x * y * s * holling_disk(o, ar_norm[1], [o, m...], ar_norm, 0.5) * e 
        rr = - fr / e

        return (fr, rr)
    end

    @rule fwm for a in (collect ∘ values)(attack_rates)

        @var s = subject(reverse_atk[a]) 
        @var o = object(reverse_atk[a])
        @var a = a
        @var m = with_role(:AF_modifier, reverse_atk[a])
        @var ar = atk_groups[a]

        @param x = traits[traits.species .== subject(reverse_atk[a]), :metabolic_rate][1]
        @param y = traits[traits.species .== object(reverse_atk[a]), :max_consumption][1]
        @param g = g

        ar_norm = ar ./ Ref(sum(ar))

        object_gain = x * y * holling_disk(o, ar_norm[1], [o, m...], ar_norm, 0.5)
        mean_gain = x * y * mean(holling_disk.([o, m...], ar_norm[1], Ref([o, m...]), Ref(ar_norm), Ref(0.5)))
        
        f = g * a * (object_gain - mean_gain)

        return min(f, 1.0)
    end

    return (traits, fwm)
end