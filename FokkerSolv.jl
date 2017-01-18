module FokkerSolv
"""
	Solve the Fokker Planck Equation including pitch angle 
	diffusion.
	The physical state of the system at time t is stored in 
	a state variable. This state is given to calculate new
	coefficients for the equation
"""

# global parameters

const N = 1<<10 			# resolution parameter in p
const Na = N				# resolution parameter in alpha
const Nt = 1<<10			# resolution parameter in time

const tbeg = 0.0			# time begin
const tend = 1.0			# time end
const dt = (tend-tbeg)/Nt	# time increment

const pmin = 1.0			# log10(p[1])
const pmax = 7.0			# log10(p[N])

const amin = 0				# alpha[1]
const amax = pi				# alpha[Na]

# constants

const c = 2.99792458e10		# speed of light
const mp = 1.6726231e-24	# proton mass
const Bcmb = 3.24516e-6 	# mag field eq. of CMB
const ad_idx = 5.0/3.0		# adiabatic index
const mean_mol_weight = 0.6	# mu

# global variables

const p = logspace(pmin, pmax, N)			# momentum p/me/c
const p_phys = p*me*c
const gamma = p - 1
const alpha = logspace(amin, amax, Na)		# pitch angle

# complex types & their constructors

type PhysicalState

	nth::Float64			# Thermal number density
	Bfld::Array{Float64}(3)	# Magnetic field
	vturb::Float64			# Turbulent velocity
	Temp::Float64			# Temperature
	cs::Float64				# speed of sound

	# inner constructor
	function PhysicalState(t::Float64, nCRe::Array{Float64}(N,Na))
	
		nth = ThermalNumberDensity(t)
		Bfld = MagneticField(t)
		Vturb = TurbulentVelocity(t, nCRe)
		Temp = Temperature(t)
		
		# derived parameters

		cs = 9.75e5*sqrt(ad_idx*Temp/mean_mol_weight)
		
		return new(nth, B, Vturb, Temp, cs)
	end
end

type FKPCoefficients

	Dpp::Array{Float64}(N,N) 	# Reacceleration Coefficient
	Hp::Array{Float64}(N,N)		# Cooling Coefficient
	Q::Array{Float64}(N,N)		# Injection Coefficient
	T::Array{Float64}(N,N)		# Escape Coefficient

	# inner constructor
	function FKPCoefficients(t::Float64, state::PhysicalState, 
						  	nCRe::Array{Float64}(N,Na)) 
		
		Dpp = ReaccelerationCoeff(t, state)
		Hp = CoolingCoeff(t, state)
		Q = InjectionCoeff(t, state)
		T = EscapeCoeff(t, state)

		return new(Dpp, Hp, Q, T)
	end
end

# Driver routine

function FKPSolv

	nCRe = InitialCReSpectrum() 				# CRe spectrum n(p,alpha)

	t = tbeg									# time [s]

	for i=1:Nt
	
		state = PhysicalState(t, nCRe) 			# get next physical state 
		coeff = FKPCoefficients(t, state, nCRe)	# get new coefficients

		# algorithm

		nCRe = 1

		t += dt
	end

end

function InitialCReSpectrum()

	nCRe = Array{Float64}(N, Na)

	nCRe[:,:] = 0 # set == 0

	return nCRe
end

# Physical State Functions

function ThermalNumberDensity(t::Float64)
	
	return 1e-3 # cm^-3
end

function MagneticField(t::Float64)

	return [1,1,1] * 1e-6 # G
end

function TurbulentVelocity(t::Float64)

	return 100e5 # cm/s
end

function Temperature(t::Float64)

	return 1e8 # K
end

# Fokker Planck Coefficients

function ReaccelerationCoeff(t::Float64, state::PhysicalStateQuantities)

	eta_t = 0.1

	Dpp = Array{Float64}(N,Na)

	Dpp[:,:] = 1.64e-8 * eta_t * state.vturb^4/state.scale/state.cs^2

	for i = 1, N
		for j = 1, Na
		
			Dpp[i,j] *= p[i]^2

		end 
	end 

	return Dpp
end

function CoolingCoeff(t::Float64, state::PhysicalStateQuantities)
	
	Hp = Array{Float64}(N,Na)

	return Hp
end

function InjectionCoeff(t::Float64, state::PhysicalStateQuantities)
	
	Q = Array{Float64}(N,Na)

	return Q
end

function EscapeCoeff(t::Float64, state::PhysicalStateQuantities)
	
	T = Array{Float64}(N,Na)

	return T
end

end # FokkerSolv
