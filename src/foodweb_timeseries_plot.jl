function foodweb_timeseries(sol, fwm)

    fig = CairoMakie.Figure()
    ax = CairoMakie.Axis(fig[1,1])
    
    for sp in species(fwm)

        # TODO Make this find every positive population interval and draw a line
        # _segment_ for it.
        ext = findfirst(x -> x == 0.0, sol[sp])
        if isnothing(ext)

            ext = length(sol[sp])
        end

        CairoMakie.lines!(ax, sol.t[1:ext], sol[sp][1:ext])
        
        # TODO Find all the upcrossings and downcrossings and use different
        # arrows to distinguish them.
        CairoMakie.scatter!(ax, sol.t[ext], -0.05, 
            marker = '↘', 
            markersize = 15)

        # TODO Optional labelling of all the invasion and extinction events with
        # the species name

    end

    return fig
end