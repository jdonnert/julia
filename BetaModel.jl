module BetaModel

export Density

function Density(r, rho0, rc, beta)

	rho0 .* (1 + r./rc)^(-3/2 .* beta)

end

end # module
