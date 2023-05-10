module InterfaceHysteris

    export OptimProb

    using ForwardDiff

    @Base.kwdef struct OptimProb

        χ :: Float64
        h :: Vector{Float64}
        mₚ :: Vector{Float64}  

        S :: Function
        U :: Function
        ∂S :: Function
        ∂U :: Function
        
        u0 :: Vector{Float64}
        m0 :: Vector{Float64}
  
        hasObjectiveHessian :: Bool = false
        ∂²S :: Function = ForwardDiff.hessian(S,u0)
        
        function OptimProb(χ,h,mₚ,S,U,∂S,∂U,u0,m0,∂²S)
            OptimProb(χ,h,mₚ,S,U,∂S,∂U,u0,m0,true,∂²S)
        end
    end
    
end