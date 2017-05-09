#module WENO5 			# Jiang & Wu 1999 JCAP 150

using Plots

const N = 256			# number of cells
const tmax = 0.2		# end of time integration

const NBnd = 3			# number of boundary cells
const iMin = 1+NBnd		# grid loop minimum
const iMax = N-NBnd		# grid loop maximum

const xsize = 1
const dx = xsize/N

const gamma = 5.0/3.0	# adiabatic index

const p = 5				# order
const courFac = 0.8		# Courant Factor

const eps = 1e-6 		# Jiang & Wu after eq 2.3 

export WENO5

function WENO5()

	q = initial_conditions() # Shock Tube
	
	t = 0
	nstep = 0

	while t < tmax 	# explicit fourth order Runge-Kutta

		vxmax = compute_global_vmax(q)

		dt = courFac*dx/vxmax

		println("$nstep : t = $t dt = $dt vmax=$vxmax")

		q0 = copy(q)

		Q0 = weno5_fluxes(q0)
		break
		q1 = q0 + 1/2 * dt/dx * Q0
		
		boundaries!(q1)

		Q1 = weno5_fluxes(q1)

		q2 = q0 + 1/2 * dt/dx * Q1

		boundaries!(q2)

		Q2 = weno5_fluxes(q2)

		q3 = q0 + dt/dx * Q2

		boundaries!(q3)

		Q3 = weno5_fluxes(q3)

		q = 1/3 * (-q0 + q1 + 2*q2 + q3 + 1/2 * dt/dx * Q3)

		boundaries!(q)

		t += dt

		nstep += 1

	end
	
	return
end

function weno5_fluxes(q::Array{Float64,2})
	
	u = compute_primitive_variables(q)

	println("$(u[1,:])")

	return
end

function compute_primitive_variables(q::Array{Float64,2})

	u = copy(q)  # [rho,Mx,My,Mz,Bx,By,Bz,S,E]
	
	u[:,2] ./= u[:,1] # vx = Mx/rho
	u[:,3] ./= u[:,1] # vy
	u[:,4] ./= u[:,1] # vz

	v2 = u[:,2].^2 + u[:,3].^2 + u[:,4].^2
	B2 = u[:,5].^2 + u[:,6].^2 + u[:,7].^2

	u[:,8] .*= u[:,1].^(gamma-1)	# Pressure from S

	u[:,9] -= 0.5 * (u[:,1].*v2 + B2) # Pressure from E
	u[:,9] .*= (gamma-1)

	return u # [rho,vx,vy,vz,Bx,By,Bz,P(S),P(E)]
end

function boundaries!(q::Array{Float64,2})

	return
end

function compute_global_vmax(q::Array{Float64,2})

	vmax = zeros(N)

	#Threads.@threads
	for i = iMin:iMax # Jiang & Wu after eq 2.24

		rho = (q[i,1] + q[i+1,1])/2
		
		vx = (q[i,2] + q[i+1,2])/2 / rho
		vy = (q[i,3] + q[i+1,3])/2 / rho
		vz = (q[i,4] + q[i+1,4])/2 / rho

		Bx = (q[i,5] + q[i+1,5])/2
		By = (q[i,6] + q[i+1,6])/2
		Bz = (q[i,7] + q[i+1,7])/2

		S = (q[i,8] + q[i+1,8])/2

		pres = S * rho^(gamma-1)

		cs2 = max(0, gamma * abs(pres/rho))
		B2 = Bx^2 + By^2 + Bz^2

		bbn2 = B2/rho
		bnx2 = Bx^2/rho

		rootx = sqrt((bbn2+cs2)^2 - 4*bnx2*cs2)

		lfx = sqrt(0.5 * (bbn2+cs2+rootx)) 

		vmax[i] = abs(vx) + lfx

	end

	return maximum(vmax)
end

function initial_conditions()

	sqrt4p = sqrt(4.0*pi)
	
	# conserved variables left & right
	# states:             rho   vx    vy   vz   Bx        By           Bz      Press
	ql = state2conserved(1.08, 1.2, 0.01, 0.5, 2/sqrt4p, 3.6/sqrt4p, 2/sqrt4p, 0.95)
	qr = state2conserved(1.0,    0,    0,   0, 2/sqrt4p, 4/sqrt4p,   2/sqrt4p, 1   )

	q0 = zeros(Float64, N, 9)

	for i = 1:N 

		if i < N/2
			q0[i,:] = ql
		else
			q0[i,:] = qr
		end
	end

	return q0
end

function state2conserved(rho, vx, vy, vz, Bx, By, Bz, P)
	
	v2 = vx^2 + vy^2 + vz^2
	B2 = Bx^2 + By^2 + Bz^2

	EE = P / (gamma-1) + 0.5 * (rho*v2+B2)
	SS = P / rho^(gamma-1)

	return [rho, rho*vx, rho*vy, rho*vz, Bx, By, Bz, SS, EE]
end

#end # module
