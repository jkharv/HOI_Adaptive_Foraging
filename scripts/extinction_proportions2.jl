include("../src/HOI_Adaptive_Foraging.jl")

using .HOI_Adaptive_Foraging
using OrdinaryDiffEq
using HigherOrderFoodwebs
using SpeciesInteractionNetworks
using ModelingToolkit
using Random
using LinearAlgebra
using Statistics
using Distributions
using DataFrames
using CSVFiles

include("my_model.jl")

function simulation_batch(s, c, b, gmin, gmax, n, fw_num)

    fwm = build_my_fwm(s, c, b, 0.0);
    prob = ODEProblem(fwm)

    stepsize = (gmax - gmin) / (n - 1)
    probs = []
 
    nz_species = 0
    for sp in species(fwm)

        if fwm.u0[fwm.conversion_dict[sp]] > 0
            nz_species += 1
            println(fwm.u0[fwm.conversion_dict[sp]])
        end
    end 

    # This creates a NaN if any batches have only species.
    # Whenever n mod batch size  = 1

    df = DataFrame(
        :foodweb => Vector{Int64}(),
        :nominal_richness => Vector{Int64}(),
        :starting_richness => Vector{Int64}(),
        :connectivity => Vector{Float64}(),
        :g => Vector{Float64}(),  
        :primary_extinctions => Vector{Int64}(),
        :secondary_extinctions => Vector{Int64}()
    )
    df_lock = ReentrantLock()    
    
    # Set simulations of the same foodweb with a bunch 
    # of different values for adaptation rate.
    for gval ∈ gmin:stepsize:gmax
   
        fwm.param_vals[fwm.conversion_dict[:g]] = gval        

        p = remake(prob, p = fwm.param_vals)
        push!(probs, (p, gval))
    end
   
    Threads.@threads for (p, gval) in probs

        # Set up vectors to record extinction data in
        primary_extinctions = Vector{Tuple{Float64,Symbol}}()
        secondary_extinctions = Vector{Tuple{Float64,Symbol}}()

        # Set up callbacks
        et = ExtinctionThresholdCallback(fwm, 1e-20; 
            extinction_history = secondary_extinctions);
        es = ExtinctionSequenceCallback(fwm, shuffle(deepcopy(species(fwm))), 500.0; 
            extinction_history = primary_extinctions);
        rt = RichnessTerminationCallback(fwm, 0.5);

        solve(p, AutoTsit5(Rosenbrock23());
            callback = CallbackSet(et, es, rt), 
            force_dtmin = true,
            save_on = false,
            maxiters = 1e6,
            tspan = (0, 500 * richness(fwm) + 500)
        );

        lock(df_lock)

        push!(df,
            (
                foodweb = fw_num,
                nominal_richness = s,
                starting_richness = nz_species,
                connectivity = c,
                g = gval,
                primary_extinctions = length(primary_extinctions),
                secondary_extinctions = length(secondary_extinctions)
            )
        )

        unlock(df_lock)
    end

    return df
end


function simulations(s, c, b, gmin, gmax, n, batch_size)


    n_batches  = div(n, batch_size)
    remainder = mod(n, batch_size) 

    itr = [[batch_size for i in 1:n_batches]..., remainder]

    df = DataFrame()

    for (i, n) in enumerate(itr)

        df_batch = simulation_batch(s, c, b, gmin, gmax, n, i)

        append!(df, df_batch)
    end

    return df
end

simulations(15, 0.3, 4, 0.0, 0.25, 10, 3)