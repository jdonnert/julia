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

export H100, CosmoPar, CC, WMAP7, Planck15
export Show, Omega_tot, H0, Hubble_Distance, Hubble_Time, Ez, Hubble, Comoving_Distance, 
		Transverse_Comoving_Distance, Angular_Diameter_Distance, Luminosity_Distance, Angular_Size, 
		Critical_Density, Overdensity_Parameter, Luminosity2Flux, Flux2Luminosity, Proper_Time_a, 
		Proper_Time_z

""" Fundamental Cosmological Parameters """
type CosmoPar
	Name :: String		# name of set
	Hbpar :: Float64  	# Hubble(z=0)/H100 parameter
	Omega_b	:: Float64	# Baryon density parameter
	Omega_M	:: Float64 	# Matter density
	Omega_L	:: Float64	# Dark energy density parameter
	Omega_r	:: Float64	# Radiation density parameter
	Sigma_8	:: Float64	# Fluctuation amplitude at 8/h Mpc
	n :: Float64		# Primordial spectral index of fluctuation power spectrum
end

const CC = CosmoPar("Concordance Cosmology", 0.7, 0.03, 0.3, 0.7, 0, 0.8, 1)
const WMAP7 = CosmoPar("(Komatsu+ 2010, Larson+ 2010)", 
					   0.702, 0.728,0.0455, 0.272,2.47e-5,0.807, 0.967)
const Planck15 = CosmoPar("(Planck Collaboration 2015 XIII)",
						  0.6774,0.6911,0.0223,0.3089,9.23e-5,0.8159,0.9667)

function Show(cp::CosmoPar; z=0)

	println("\nFor '$(cp.Name)' cosmology :")
    println("   Hbpar    	= $(cp.Hbpar)")
    println("   Omega_L  	= $(cp.Omega_L)")
    println("   Omega_M  	= $(cp.Omega_M)")
    println("   Omega_b  	= $(cp.Omega_b)")
    println("   Omega_r  	= $(cp.Omega_r)")
    println("   Sigma_8  	= $(cp.Sigma_8)")
    println("   n       	= $(cp.n)")
	
	println("\nDerived constants : " )
	println("   Omega_tot 	    = $(Omega_tot(cp))")
	println("   Omega_k 	    = $(Omega_k(cp))")
	println("   H0       	    = $(H0(cp)) 1/s")
	println("   Hubble_Distance = $(Hubble_Distance(cp)) cm")
	println("   Hubble_Time     = $(Hubble_Time(cp)) s")
	
	println("\nAt redshift z = $z : " )
	println("   E(z)                         = $(Ez(cp, z))")
	println("   Hubble(z)                    = $(Hubble(cp, z)) 1/s")
	println("   Comoving_Distance            = $(Comoving_Distance(cp, z)) cm")
	println("   Transverse_Comoving_Distance = $(Transverse_Comoving_Distance(cp, z)) cm")
	println("   Angular_Diameter_Distance    = $(Angular_Diameter_Distance(cp, z)) cm")
	println("   Luminosity_Distance          = $(Luminosity_Distance(cp, z)) cm")
	println("   Angular_Size                 = $(Angular_Size(cp, 1, z)) arcmin/kpc")
	println("   Critical_Density             = $(Critical_Density(cp, z)) g/cm^3")
	println("   Overdensity_Parameter        = $(Overdensity_Parameter(cp, z))")
	println("   Luminosity2Flux              = $(Luminosity2Flux(cp, 1, z)) erg/s/cm^2")
	println("   Flux2Luminosity              = $(Flux2Luminosity(cp, 1, z)) erg/s")
	println("   Proper_Time                  = $(Proper_Time_z(cp, z)) s")

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

function Hubble_Distance(cp::CosmoPar)
	
	return c/H0(cp)
end

function Hubble_Time(cp::CosmoPar)
	
	t_Hubble = 1/H0(cp)
end

# functions that depend on redshift

""" E(z)  Mo+ 2010 eq. 2.62, 3.75, Boehringer+ 2012 eq 5 """
function Ez(cp::CosmoPar, z) 

	return sqrt(cp.Omega_L + cp.Omega_M*(1+z)^3 
			 + cp.Omega_r*(1+z)^4 + (1 - Omega_tot(cp))*(1+z)^2) 
end

""" Redshift dependent Hubble parameter (Mo+ 2010 eq. 3.74) """
function Hubble(cp::CosmoPar, z) 
	
	return H0(cp)/ Ez(cp, z) 
end

""" Mo+ 2010 eq. 3.102 """
function Comoving_Distance(cp::CosmoPar, z) 
	
	global int_cp = cp # global var holds parameters for the integrant
	
	int, relErr = quadgk(int_dComov, 0, z)

	return c/H0(cp) * int
end

function int_dComov(z) # integrant

	return 1/Ez(int_cp, z)
end

function Transverse_Comoving_Distance(cp::CosmoPar, z)

	dHubble = Hubble_Distance(cp)
	dComov = Comoving_Distance(cp, z)
	sqrtOk = sqrt(abs(Omega_k(cp)))

	if Omega_k(cp) == 0
		result = dComov
	elseif Omega_k(cp) > 0
		result = dHubble/sqrtOk * sinh(sqrtOk * dComov/dHubble)
	else
		result = dHubble/sqrtOk * sin(sqrtOk * dComov/dHubble)
	end

	return result
end

function Angular_Diameter_Distance(cp::CosmoPar, z)
	
	return Transverse_Comoving_Distance(cp, z) / (1+z)
end

function Luminosity_Distance(cp::CosmoPar, z)
	
	return (1+z) * Transverse_Comoving_Distance(cp, z)
end

""" Convert kpc to arcmin on the sky (kpc2arcmin) """
function Angular_Size(cp::CosmoPar, nkpc, z)

	return nkpc*kpc2cm / Angular_Diameter_Distance(cp, z) / arcmin2rad
end

function Critical_Density(cp::CosmoPar, z)

	return (cp.Omega_L + (1+z)^3*cp.Omega_M) * 3 * H0(cp)^2 / (8*pi*grav)
end

""" Delta(z) (Pierpaoli+ 01)"""
function Overdensity_Parameter(cp::CosmoPar, z)

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
			sum += cij[i][j] * x^(i-1) * y^(j-1)
		end
	end

	return cp.Omega_M*(1+z)^3 * sum
end


""" Convert luminosity to flux in cgs"""
function Luminosity2Flux(cp::CosmoPar, L, z)

	return L / (4*pi*Luminosity_Distance(cp, z)^2)
end

""" Convert Flux to luminosity in cgs """
function Flux2Luminosity(cp::CosmoPar, F, z)

	return F * 4*pi*Luminosity_Distance(cp, z)^2
end

""" Expansion factor to age of the Universe in s """
function Proper_Time_a(cp::CosmoPar, a)

	z = 1/a - 1

	return Proper_Time_z(cp, z)
end

""" Age of the Universe in s from redshift (Mo+ 2010 eq. 3.94) """
function Proper_Time_z(cp::CosmoPar, z)

	global int_cp = cp # parameters of the integrant

	result = quadgk(int_dComov, 0, z, reltol=1e-10, order=10)

	return 1/H0(cp) * result[1]
end

function int_z2t(z) # integrant
	return 1/Ez(int_cp, z)/(1+z)
end

end # module Cosmology
