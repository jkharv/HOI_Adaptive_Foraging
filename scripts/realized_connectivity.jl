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
using CSV

include("my_model.jl")

empty_df() = DataFrame(
        :foodweb => Vector{Int64}(),
        :nominal_richness => Vector{Int64}(),
        :starting_richness => Vector{Int64}(),
        :nominal_c => Vector{Float64}(),
        :g => Vector{Float64}(),  
        :realized_c => Vector{Float64}()
)

function simulation_batch(s, c, b, gmin, gmax, n, fw_num)

    fwm = build_my_fwm(s, c, b, 0.4);
    prob = ODEProblem(fwm)

    stepsize = (gmax - gmin) / (n - 1)
    probs = []
 
    nz_species = 0
    for sp in species(fwm)

        if fwm.u0[sym_to_var(fwm, sp)] > 0
            nz_species += 1
        end
    end 

    # This creates a NaN if any batches have only species.
    # Whenever n mod batch size  = 1

    df = empty_df()
    df_lock = ReentrantLock()    
    
    # Set simulations of the same foodweb with a bunch 
    # of different values for adaptation rate.
    for gval ∈ gmin:stepsize:gmax

        # Not generally true, gotta finish the move of params into `FoodwebVariables` too
        g = fwm.params[1]        

        fwm.param_vals[g] = gval        

        p = remake(prob, p = fwm.param_vals)
        push!(probs, (p, gval))
    end
   
    Threads.@threads for (p, gval) in probs

        et = ExtinctionThresholdCallback(fwm, 1e-20);

        sol = solve(p, AutoTsit5(Rosenbrock23());
            callback = et, 
            force_dtmin = true,
            maxiters = 1e6,
            tspan = (0, 2000)
        );

        lock(df_lock)
        
        intxs = trophic_flux(fwm, sol, 2000; include_loops = false)
        rc_val = count(!iszero, values(intxs)) / 10^2

        push!(df,
            (
                foodweb = fw_num,
                nominal_richness = s,
                starting_richness = nz_species,
                nominal_c = c,
                g = gval,
                realized_c = rc_val
            )
        )

        unlock(df_lock)
    end

    return df
end

function simulations(s, c, b, gmin, gmax, n, batch_size, outpath)

    # Create the output file. Replacing whatever was there before.
    CSV.write(outpath, empty_df(); header = true, append = false)

    n_batches  = div(n, batch_size)
    remainder = mod(n, batch_size) 

    itr = [[batch_size for i in 1:n_batches]..., remainder]

    df = empty_df()

    for (i, n) in enumerate(itr)

        df_batch = simulation_batch(s, c, b, gmin, gmax, n, i)

        append!(df, df_batch)
        CSV.write(outpath, df_batch; append = true)
    end

    return df
end

simulations(5, 0.3, 1, 0.0, 0.25, 20, 5, "data.csv")