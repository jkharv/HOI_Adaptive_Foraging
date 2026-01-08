function rectangular_web(width::Int64, height::Int64, connectance)

    spp   = Vector{Vector{Symbol}}(undef, height) 
    intxs = Vector{AnnotatedHyperedge{Symbol}}() 

    spp[1] = [Symbol("sp_1_$(i)") for i in 1:width]

    for tl in 2:height
        
        spp[tl] = [Symbol("sp_$(tl)_$(i)") for i in 1:width]
        
        pairs = (collect ∘ Iterators.product)(spp[tl - 1], spp[tl])

        for (r, p) in pairs 

            if rand() < connectance            

                i = AnnotatedHyperedge([r, p], [:object, :subject])            
                push!(intxs, i)
            end
        end
    end

    spp = (Unipartite ∘ collect ∘ Iterators.flatten)(spp) 

    return AnnotatedHypergraph(spp, intxs)
end