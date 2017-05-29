#module Weno5_2D

using LaTeXStrings

const tmax = 0.1		# end of time integration

const NBnd = 3			# number of boundary cells

const N = 256 + 2*NBnd	# number of cells including boundaries
const M = 64 + 2*NBnd	# number of cells including boundaries

const iMin = 1 + NBnd	# grid loop minimum without boundaries
const iMax = N - NBnd	# grid loop maximum without boundaries
const jMin = 1 + NBnd	# grid loop minimum without boundaries
const jMax = N - NBnd	# grid loop maximum without boundaries

const xsize = 4
const dx = xsize/(N - 2*NBnd)
const ysize = 1
const dy = ysize/(N - 2*NBnd)

const gamma = 5.0/3.0	# adiabatic index & friends
const gam0 = 1-gamma
const gam1 = (gamma-1)/2
const gam2 = (gamma-2)/(gamma-1)
const gamS = (gamma-1)/gamma

const courFac = 0.8		# Courant Factor

const eps = 1e-6 		# Jiang & Wu eq 2.3++ 

export WENO5_2D

function WENO5_2D()


	return

end # WENO5_2D


function ES_Switch(q_E::Array{Float64,3}, q_S::Array{Float64,3})
	
	q_SE = copy(q_E)
	q_SE[:,:,8] = E2S(q_E) 		# convert q(E) to q(S(E))

	dS = abs(q_SE[:,:,8] - q_S[:,:,8])

	bad = find(dS .>= 1e-5)

	q = zeros(N, 9)
	q[:,:,1:8] = q_S			# use q(S) by default
	q[bad,1:8] = q_SE[bad,:]  	# replace q at the bad locations
	q[:,:,9] = S2E(q)

	return q, bad
end


function E2S(qE::Array{Float64,3})

	E = copy(qE[:,:,8])
	rho = qE[:,:,1]
	mom2 = qE[:,:,2].^2 + qE[:,:,3].^2 + qE[:,:,4].^2
	B2 = qE[:,:,5].^2 + qE[:,:,6].^2 + qE[:,:,7].^2
	
	S = (E - 0.5*mom2./rho - B2/2)*(gamma-1)./rho.^(gamma-1)
	
	return S  # Ryu+ 1993, eq. 2.3 + 1
end

function S2E(qS::Array{Float64,3})
	
	S = qS[:,:,8]
	rho = qS[:,:,1]
	mom2 = qS[:,:,2].^2 + qS[:,:,3].^2 + qS[:,:,4].^2
	B2 = qS[:,:,5].^2 + qS[:,:,6].^2 + qS[:,:,7].^2

	E = rho.^(gamma-1).*S/(gamma-1) + B2/2 + 0.5.*mom2./rho

	return E # Ryu+ 1993, eq. 2.3+1
end

function compute_global_vmax(q::Array{Float64,3})

	vxmax = Array(Float64, N,M)
	vymax = Array(Float64, N,M)

	
	Threads.@threads for j = jMin:jMax # Jiang & Wu after eq 2.24
		for i = iMin:iMax
			
			rho = (q[i,j,1] + q[i+1,j,1])/2
		
			vx = (q[i,j,2] + q[i+1,j,2])/2 / rho
			vy = (q[i,j,3] + q[i+1,j,3])/2 / rho
			vz = (q[i,j,4] + q[i+1,j,4])/2 / rho

			Bx = (q[i,j,5] + q[i+1,j,5])/2
			By = (q[i,j,6] + q[i+1,j,6])/2
			Bz = (q[i,j,7] + q[i+1,j,7])/2

			S = (q[i,j,8] + q[i+1,j,8])/2

			pres = S * rho^(gamma-1)

			cs2 = gamma * abs(pres/rho)
			B2 = Bx^2 + By^2 + Bz^2

			bbn2 = B2/rho

			bnx2 = Bx^2/rho
			bny2 = By^2/rho

			rootx = sqrt((bbn2+cs2)^2 - 4*bnx2*cs2)
			rooty = sqrt((bbn2+cs2)^2 - 4*bny2*cs2)

			lfx = sqrt(0.5 * (bbn2+cs2+rootx)) 
			lfy = sqrt(0.5 * (bbn2+cs2+rooty)) 

			vxmax[i,j] = abs(vx) + lfx
			vymax[i,j] = abs(vy) + lfy
		end
	end

	return maximum(vxmax), maximum(vymax)
end

function initial_conditions() # Shock Cloud interaction

	const R0 = 0.12
	const R1 = 0.19

	q = Array(Float64, N, M, 9)

	# state variables left & right
	#        rho      vx    vy vz Bx        By      Bz    Press
	sl = [2.11820, 13.69836, 0, 0, 0, 19.33647, 19.33647, 65.8881]
	sr = [      1,        0, 0, 0, 0,  9.12871, 9.12871,      1.0]

	for j = 1:M 
		for i = 1:N

			x = i * dx
			y = j * dy

			rr = sqrt((x - 1.5)^2 + (y - 0.5)^2)

			if x <= 1.2
				s = sl
			else 
				s = sr
			end

			if rr <= R0
				s[1] = 10
			elseif rr > R0 && rr <= R1 
				s[1] = 1 + 9 * (R1-rr) / (R1-R0)
			end

			q[i,j,:] = state2conserved(s[1],s[2],s[3],s[4],s[5],s[6],s[7],s[8])

		end
	end


	return q
end


function state2conserved(rho, vx, vy, vz, Bx, By, Bz, P)
	
	v2 = vx^2 + vy^2 + vz^2
	B2 = Bx^2 + By^2 + Bz^2

	E = P / (gamma-1) + 0.5 * (rho*v2+B2)
	S = P / rho^(gamma-1)

	return [rho, rho*vx, rho*vy, rho*vz, Bx, By, Bz, S, E]
end

function safe_sqrt(x::Array{Float64,2})
	
	y = copy(x)
	bad = find(y .< 0)
	y[bad] = 0
		
	return sqrt(y)
end


#end # module
