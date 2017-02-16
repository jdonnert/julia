"""
Implement a type setting fundamental cosmological parameters (CosmoPar) and a bunch of methods that 
compute cosmological quantities from it. We provide a number of standard parameter 
sets: Concord, WMAP7, Planck15

	using Cosmology

	cp = Concord		# copy concordance cosmology
	println(dHubble(cp))

	cp.Hbpar = 71.5 	# change Hubble parameter
	println(dHubble(cp))

	Show(cp)

Mo, van den Bosch & White 2010, Boehringer et al. 2012

Feb. 2017, Initial Version (Julius Donnert, UMN)
"""

module Cosmology

importall CGSUnits

export H100, CosmoPar, Concord, WMAP7, Planck15
export Show, Omega_tot, H0, dHubble, tHubble, Ez, Hubble, dComoving, dTransComoving, 
		dAngular, dLuminosity, arcmin2kpc, kpc2arcmin, rhoCrit, overdensParam, lum2flux, 
		flux2lum, t2a, a2t, z2t, t2z

""" Fundamental Cosmological Parameters """
type CosmoPar
	Name :: String		# name of set
	Hbpar :: Float64  	# Hubble(z=0)/H100 parameter
	Omega_b	:: Float64	# Baryon density parameter
	Omega_M	:: Float64 	# Matter density
	Omega_L	:: Float64	# Dark energy density parameter
	Omega_r	:: Float64	# Dark energy density parameter
	Sigma_8	:: Float64	# Fluctuation amplitude at 8/h Mpc
	n :: Float64		# Primordial spectral index of fluctuation power spectrum
end

const Concord = CosmoPar("Concordance", 0.7, 0.03, 0.3, 0.7, 0, 0.8, 1)
const WMAP7 = CosmoPar("(Komatsu+ 2010, Larson+ 2010)", 0.702, 0.728,0.0455, 0.272,2.47e-5,0.807, 0.967)
const Planck15 = CosmoPar("(Planck Collaboration 2015 XIII)",0.6774,0.6911,0.0223,0.3089,9.23e-5,0.8159,0.9667)

function Show(cp::CosmoPar; z=0)

	println("\nFor $(cp.Name) cosmology :")
    println("   Hbpar    	= $(cp.Hbpar)")
    println("   Omega_L  	= $(cp.Omega_L)")
    println("   Omega_M  	= $(cp.Omega_M)")
    println("   Omega_b  	= $(cp.Omega_b)")
    println("   Omega_r  	= $(cp.Omega_r)")
    println("   Sigma_8  	= $(cp.Sigma_8)")
    println("   n       	= $(cp.n)")
	
	println("\nDerived constants : " )
	println("   Omega_tot 	= $(Omega_tot(cp))")
	println("   Omega_k 	= $(Omega_k(cp))")
	println("   H0       	= $(H0(cp)) 1/s")
	println("   dHubble 	= $(dHubble(cp)) cm")
	println("   tHubble  	= $(tHubble(cp)) cm")
	
	println("\nAt redshift z = $z : " )
	println("   E(z)			= $(Ez(cp, z))")
	println("   Hubble(z)		= $(Hubble(cp, z)) 1/s")
	println("   dComoving		= $(dComoving(cp, z)) cm")
	println("   dTransComoving	= $(dTransComoving(cp, z)) cm")
	println("   dAngular		= $(dAngular(cp, z)) cm")
	println("   dLuminosity		= $(dLuminosity(cp, z)) cm")
	println("   arcmin2kpc		= $(arcmin2kpc(cp, 1, z)) kpc")
	println("   kpc2arcmin		= $(kpc2arcmin(cp, 1, z)) arcmin")
	println("   rhoCrit      	= $(rhoCrit(cp, z)) g/cm^3")
	println("   overdensParam	= $(overdensParam(cp, z))")
	println("   lum2flux		= $(lum2flux(cp, 1, z)) erg/s/cm^2")
	println("   flux2lum		= $(flux2lum(cp, 1, z)) erg/s")

end

# derived parameters

""" Total Density Parameter """
function Omega_tot(cp::CosmoPar)
	
	return cp.Omega_M + cp.Omega_L + cp.Omega_r
end

""" Curvature Density """
function Omega_k(cp::CosmoPar)
	
	return 1 - cp.Omega_M - cp.Omega_L
end

""" Hubble Constant at z=0 in cgs """
function H0(cp::CosmoPar)
	
	return cp.Hbpar * H100
end

""" Hubble Distance (Horizon) """
function dHubble(cp::CosmoPar)
	
	return c/H0(cp)
end

""" Hubble Time (Age of the Universe) """
function tHubble(cp::CosmoPar)
	
	t_Hubble = 1/H0(cp)
end

# functions that take arguments

""" E(z)  Mo+ 2010 eq. 2.62, 3.75, Boehringer+ 2012 eq 5 """
function Ez(cp::CosmoPar, z) 

	return sqrt(cp.Omega_L + cp.Omega_M*(1+z)^3 
			 + cp.Omega_r*(1+z)^4 + (1 - Omega_tot(cp))*(1+z)^2) 
end

""" Hubble Constant at redshift z, (Mo+ 2010 eq. 3.74)"""
function Hubble(cp::CosmoPar, z) 
	
	return H0(cp)/ Ez(cp, z) 
end


""" comoving distance at redshift z (Wickramasinghe+ 2010) """
function dComoving(cp::CosmoPar, z) 
	
	@assert((Omega_k(cp) == 0) && (cp.Omega_L != 0), 
			"Approximation for comoving distance not valid")

	alpha = 1 + 2*cp.Omega_L/ (1 - cp.Omega_L) / (1+z)^3
    x = log(alpha + sqrt(alpha^2 - 1))
    Sigma_z = 3 * x^(1/3) * 2^(2/3) * (1 - x^2/252. + x^4/21060)

    alpha =  1 + 2*cp.Omega_L/ (1 - cp.Omega_L)
    x = log(alpha + sqrt(alpha^2 - 1))
    Sigma_0 = 3 * x^(1/3) * 2^(2/3) * (1 - x^2/252. + x^4/21060)

	prefac = c/(3*H0(cp))/(cp.Omega_L^(1/6)*(1-cp.Omega_L)^(1/3))

	return prefac * (Sigma_0 - Sigma_z)
end

""" Transverse comoving distance at redshift z """
function dTransComoving(cp::CosmoPar, z)

	dComov = dComoving(cp, z)

	if Omega_k(cp) == 0
		return dComov
	end
	
	sqrtOk = sqrt(abs(Omega_k(cp)))
	dHubble = dHubble(cp)

	if Omega_k(cp) > 0
		return dHubble/sqrtOk * sinh(sqrtOk * dComov/dHubble)
	end

	return dHubble/sqrtOk * sin(sqrtOk * dComov/dHubble)
end

""" Angular diameter distance at redshift z """
function dAngular(cp::CosmoPar, z) # angular diameter distance
	
	return dTransComoving(cp, z) / (1+z)
end

""" Luminosity distance at redshift z """
function dLuminosity(cp::CosmoPar, z) # luminosity distance
	
	return (1+z) * dTransComoving(cp, z)
end

""" Convert arcmin to kpc on the sky """
function arcmin2kpc(cp::CosmoPar, nArcmin, z)
	
	return nArcmin * arcmin2rad * dAngular(cp, z) / kpc2cm
end

""" Convert kpc to arcmin on the sky """
function kpc2arcmin(cp::CosmoPar, nkpc, z)

	return nkpc*kpc2cm / dAngular(cp, z) / arcmin2rad
end

""" Critical Density """
function rhoCrit(cp::CosmoPar, z)

	return Omega_tot(cp) * 3 * Hubble(cp,z)^2 / (8*pi*grav)
end

""" Overdensity parameter (Pierpaoli+ 01)"""
function overdensParam(cp::CosmoPar, z)

	cij = [ [546.67, -137.82, 94.083, -204.68,  111.51], 	
	    	[-1745.6, 627.22,  -1175.2, 2445.7,  -1341.7], 	
	    	[3928.8, -1519.3, 4015.8,  -8415.3, 4642.1],	
			[-4384.8, 1748.7,  -5362.1, 11257.,  -6218.2], 	
			[1842.3, -765.53, 2507.7, -5210.7, 2867.5] ]

	x = cp.Omega_M*(1+z)^3 - 0.2
	y = cp.Omega_L

	sum = 0

	for i = 1:5
		for j = 1:5 
			@printf("	%g		%g		%g \n", cij[i][j] , x^(i-1) , y^(j-1))
			sum += cij[i][j] * x^(i-1) * y^(j-1)
		end
	end

	return cp.Omega_M*(1+z)^3 * sum
end


""" Convert luminosity to flux in cgs"""
function lum2flux(cp::CosmoPar, L, z)

	return L / (4*pi*dLuminosity(cp, z)^2)
end

""" Convert Flux to luminosity in cgs """
function flux2lum(cp::CosmoPar, F, z)

	return F * 4*pi*dLuminosity(cp, z)^2
end

function t2a(cp::CosmoPar, t)

	a = 1
end

function z2t(cp::CosmoPar, z)

	t = 1 
end

function t2z(cp::CosmoPar, t)

	z = 1
end

end # module Cosmology
