#module WENO5 			# Jiang & Wu 1999 JCAP 150

#using GR
using Plots
using ColorBrewer
using LaTeXStrings

const tmax = 0.2		# end of time integration

const NBnd = 3			# number of boundary cells

const N = 256 + 2*NBnd	# number of cells including boundaries

const iMin = 1 + NBnd	# grid loop minimum without boundaries
const iMax = N - NBnd	# grid loop maximum without boundaries

const xsize = 1
const dx = xsize/(N - 2*NBnd)

const gamma = 5.0/3.0	# adiabatic index & friends
const gam0 = 1-gamma
const gam1 = (gamma-1)/2
const gam2 = (gamma-2)/(gamma-1)
const gamS = (gamma-1)/gamma

const courFac = 0.8		# Courant Factor

const eps = 1e-6 		# Jiang & Wu eq 2.3++ 

export WENO5_1D

function WENO5_1D()

	q = initial_conditions() # [rho,Mx,My,Mz,Bx,By,Bz,S,E]
	
	x = linspace(0,xsize,N)

	t = 0
	nstep = 0

	#anim = Animation()

	#inline("mov")

	while t < tmax 	# explicit fourth order Runge-Kutta (Shu & Osher 1988)

		vxmax = compute_global_vmax(q)

		dt = courFac*dx/vxmax

		println("$nstep : t = $t dt = $dt vmax=$vxmax")
		
		q, idx = classical_RK4_step(q,dt)

	   #println("q[1]       : $(q[1,:])")
	   #println("q[130]     : $(q[130,:])")
	   #println("q[131]     : $(q[131,:])")
	   #println("q[132]     : $(q[132,:])")
	   #println("q[133]     : $(q[133,:])")
	   #println("q[iMax+3]  : $(q[iMax+3,:])")
	   #println("Replaced $(size(idx,1)):  $idx")
	   #println("===========================================")

		t += dt

		nstep += 1

		filename = "./frames/"*itoa(nstep,3)*".pdf"

		table = readdlm("output_"*itoa(nstep,3))

		#clearws()

		ymin = [0.9, -0.1, -0.3 , -0.2, 0.4, 0.9, 0.3, 0.8 ]
		ymax = [1.7,  1.4,  0.4,   0.9, 0.6, 1.7, 0.9, 1.6 ]
		ylab = ["Density","Vx","Vy","Vz","Bx","By","Bz","S"]

		for i=1:8
		
			#subplot(3, 3, i)

			#setmarkersize(0.3)
	
			plot(x, q[:,i],	"b",
				 x, table[:,i], "gd",
				 x[idx], q[idx,i],"rd"; 
				ylim=[ymin[i],ymax[i]],xlim=[0,1], xlabel="x", 
				ylabel=ylab[i])

			s = @sprintf("step %d, t = %5.3f" , nstep, t)
			textext(0.75,0.1, s)
	
		end
	
		#updatews()

#	plot(x, q[:,:], layout=8, legend=false,xlim=(0,1),ylim=(-0.3,2))
#	scatter!(x[idx], q[idx,:], markershape = :circle, markersize = 2, markercolor = :red)
#	frame(anim)

	end

	#GR.show()

	#gif(anim, "test.gif", fps=15)
	
	return
end

function classical_RK4_step(q::Array{Float64,2},dt::Float64)

		q0_E = copy(q[:,1:8])
		q0_E[:,8] = q[:,9] # 8th component is E here !
		Q0_E = weno5_flux_difference_E(q0_E)
		q1_E = q0_E - 1/2 * dt/dx * Q0_E

        q0_S = copy(q[:,1:8]) 	# 8th component remains S
        Q0_S = weno5_flux_difference_S(q0_S)
        q1_S = q0_S - 1/2 * dt/dx * Q0_S
        
		q1, idx = ES_Switch(q1_E, q1_S)

		boundaries!(q1)

		q1_E = copy(q1[:,1:8])
		q1_E[:,8] = q1[:,9]
		Q1_E = weno5_flux_difference_E(q1_E)
		q2_E = q0_E - 1/2 * dt/dx * Q1_E

		q1_S = copy(q1[:,1:8])
		Q1_S = weno5_flux_difference_S(q1_S)
		q2_S = q0_S - 1/2 * dt/dx * Q1_S

		q2, idx = ES_Switch(q2_E, q2_S)

		boundaries!(q2)

		q2_E = copy(q2[:,1:8])
		q2_E[:,8] = q2[:,9]
		Q2_E = weno5_flux_difference_E(q2_E)
		q3_E = q0_E - dt/dx * Q2_E

		q2_S = copy(q2[:,1:8])
		Q2_S = weno5_flux_difference_S(q2_S)
		q3_S = q0_S - dt/dx * Q2_S

		q3, idx = ES_Switch(q3_E, q3_S)

		boundaries!(q3)

		q3_E = copy(q3[:,1:8])
		q3_E[:,8] = q3[:,9]
		Q3_E = weno5_flux_difference_E(q3_E)
		q4_E = 1/3 * (-q0_E + q1_E + 2*q2_E + q3_E - 1/2 * dt/dx * Q3_E)

		q3_S = copy(q3[:,1:8])
		Q3_S = weno5_flux_difference_S(q3_S)
		q4_S = 1/3 * (-q0_S + q1_S + 2*q2_S + q3_S - 1/2 * dt/dx * Q3_S)

		q4, idx = ES_Switch(q4_E, q4_S)

		boundaries!(q4)

	return q4, idx
end

function ES_Switch(q_E::Array{Float64,2}, q_S::Array{Float64,2})
	
	q_SE = copy(q_E)
	q_SE[:,8] = E2S(q_E) 		# convert q(E) to q(S(E))

	dS = abs(q_SE[:,8] - q_S[:,8])

	bad = find(dS .>= 1e-5)

	q = zeros(N, 9)
	q[:,1:8] = q_S				# use q(S) by default
	q[bad,1:8] = q_SE[bad,:]  	# replace q at the bad locations
	q[:,9] = S2E(q)

	return q, bad
end


function E2S(qE::Array{Float64,2})

	E = copy(qE[:,8])
	rho = qE[:,1]
	mom2 = qE[:,2].^2 + qE[:,3].^2 + qE[:,4].^2
	B2 = qE[:,5].^2 + qE[:,6].^2 + qE[:,7].^2
	
	S = (E - 0.5*mom2./rho - B2/2)*(gamma-1)./rho.^(gamma-1)
	
	return S  # Ryu+ 1993, eq. 2.3 + 1
end

function S2E(qS::Array{Float64,2})
	
	S = qS[:,8]
	rho = qS[:,1]
	mom2 = qS[:,2].^2 + qS[:,3].^2 + qS[:,4].^2
	B2 = qS[:,5].^2 + qS[:,6].^2 + qS[:,7].^2

	E = rho.^(gamma-1).*S/(gamma-1) + B2/2 + 0.5.*mom2./rho

	return E # Ryu+ 1993, eq. 2.3+1
end

function weno5_interpolation(q::Array{Float64,2}, a::Array{Float64,2},
							 F::Array{Float64,2}, L::Array{Float64,3},
							 R::Array{Float64,3})

	Fs = zeros(Float64, 7)
	Fsk = zeros(Float64, 6)
	qsk = zeros(Float64, 6)
	dFsk = zeros(Float64, 5)
	dqsk = zeros(Float64, 5)

	dF = zeros(Float64, N, 7)

	for i=iMin-1:iMax
		
		for m=1:7

			amax = max(abs(a[i,m]),abs(a[i+1,m])) # J&W eq 2.10

			for ks=1:6 # stencil i-2 -> i+3

				Fsk[ks] = L[i,m,1]*F[i-3+ks,1] + L[i,m,2]*F[i-3+ks,2] +
                     	  L[i,m,3]*F[i-3+ks,3] + L[i,m,4]*F[i-3+ks,4] +
                     	  L[i,m,5]*F[i-3+ks,5] + L[i,m,6]*F[i-3+ks,6] +
                     	  L[i,m,7]*F[i-3+ks,7]                     
                                                                   
            	qsk[ks] = L[i,m,1]*q[i-3+ks,1] + L[i,m,2]*q[i-3+ks,2] +
                     	  L[i,m,3]*q[i-3+ks,3] + L[i,m,4]*q[i-3+ks,4] +
                     	  L[i,m,5]*q[i-3+ks,6] + L[i,m,6]*q[i-3+ks,7] +
                     	  L[i,m,7]*q[i-3+ks,8]
			end # ks

			for ks=1:5
				dFsk[ks] = Fsk[ks+1] - Fsk[ks]
				dqsk[ks] = qsk[ks+1] - qsk[ks]
			end

			first = (-Fsk[2]+7*Fsk[3]+7*Fsk[4]-Fsk[5]) / 12 # J&W eq 2.11

			aterm = (dFsk[1] + amax*dqsk[1]) / 2 # Lax-Friedrichs J&W eq. 2.10 & 2.16+
            bterm = (dFsk[2] + amax*dqsk[2]) / 2
            cterm = (dFsk[3] + amax*dqsk[3]) / 2
            dterm = (dFsk[4] + amax*dqsk[4]) / 2

            IS0 = 13*(aterm-bterm)^2 + 3*(aterm-3*bterm)^2
            IS1 = 13*(bterm-cterm)^2 + 3*(bterm+cterm)^2
            IS2 = 13*(cterm-dterm)^2 + 3*(3*cterm-dterm)^2

            alpha0 = 1/(eps+IS0)^2
            alpha1 = 6/(eps+IS1)^2
            alpha2 = 3/(eps+IS2)^2

            omega0 = alpha0/(alpha0+alpha1+alpha2)
            omega2 = alpha2/(alpha0+alpha1+alpha2)

            second = omega0*(aterm - 2*bterm + cterm)/3 + # phi_N(f+), J&W eq 2.3 + 1
                  	(omega2-0.5)*(bterm - 2*cterm + dterm)/6

            aterm = (dFsk[5] - amax*dqsk[5]) / 2 # Lax-Friedrichs J&W eq. 2.10 & 2.16+
            bterm = (dFsk[4] - amax*dqsk[4]) / 2
            cterm = (dFsk[3] - amax*dqsk[3]) / 2  
            dterm = (dFsk[2] - amax*dqsk[2]) / 2

            IS0 = 13*(aterm-bterm)^2 + 3*(aterm - 3*bterm)^2
            IS1 = 13*(bterm-cterm)^2 + 3*(bterm + cterm)^2
            IS2 = 13*(cterm-dterm)^2 + 3*(3*cterm - dterm)^2

            alpha0 = 1/(eps + IS0)^2
            alpha1 = 6/(eps + IS1)^2
            alpha2 = 3/(eps + IS2)^2

            omega0 = alpha0/(alpha0 + alpha1 + alpha2)
            omega2 = alpha2/(alpha0 + alpha1 + alpha2)

            third  = omega0*(aterm - 2*bterm + cterm) / 3 +		# phi_N(f-)
                    (omega2 - 0.5)*(bterm - 2*cterm + dterm) / 6

            Fs[m] = first - second + third # J&W eq. 2.16 + 1
		end # m
	
		for m=1:7
			
			dF[i,m] = Fs[1]*R[i,m,1] + Fs[2]*R[i,m,2] + # J&W eq. 2.17
                   	  Fs[3]*R[i,m,3] + Fs[4]*R[i,m,4] +
                      Fs[5]*R[i,m,5] + Fs[6]*R[i,m,6] +
                      Fs[7]*R[i,m,7]
		end

	end # i

	return dF
end # weno5_interpolation()

function compute_eigenvalues(u::Array{Float64,2})

	a = zeros(Float64, N, 7) # eigenvalues cell center

	B2  = u[:,5].^2 + u[:,6].^2 + u[:,7].^2
	vdB = u[:,2].*u[:,5] + u[:,3].*u[:,6] + u[:,4].*u[:,7]

	cs2 = gamma * abs(u[:,8]./u[:,1])
	bad = find(cs2 .< 0)
	cs2[bad] = 0

	bbn2 = B2./u[:,1]
	bnx2 = u[:,5].^2./u[:,1]

	root = safe_sqrt((bbn2+cs2).^2 - 4*bnx2.*cs2)

	lf = safe_sqrt((bbn2+cs2+root)/2)
	la = safe_sqrt(bnx2)
	ls = safe_sqrt((bbn2+cs2-root)/2)

	a[:,1] = u[:,2] - lf 
	a[:,2] = u[:,2] - la
	a[:,3] = u[:,2] - ls
	a[:,4] = u[:,2]
	a[:,5] = u[:,2] + ls
	a[:,6] = u[:,2] + la
	a[:,7] = u[:,2] + lf

	return a #  J&W 2.27+1

end

# SS - Code

function weno5_flux_difference_S(q::Array{Float64,2})
		
	u = compute_primitive_variables_S(q) # [rho,vx,vy,vz,Bx,By,Bz,P(S)]  

	a = compute_eigenvalues(u)

	F = compute_fluxes_S(q,u)

	L, R = compute_eigenvectors_S(q,u,F)
	
	dF = weno5_interpolation(q,a,F,L,R)
	
	Q = zeros(Float64, N, 8)

	@inbounds @simd for i=iMin:iMax
		Q[i,1] = dF[i,1] - dF[i-1,1]
		Q[i,2] = dF[i,2] - dF[i-1,2]
		Q[i,3] = dF[i,3] - dF[i-1,3]
		Q[i,4] = dF[i,4] - dF[i-1,4]
		Q[i,5] = 0
		Q[i,6] = dF[i,5] - dF[i-1,5]
		Q[i,7] = dF[i,6] - dF[i-1,6]
		Q[i,8] = dF[i,7] - dF[i-1,7]
	end

	return Q
end

function compute_eigenvectors_S(q::Array{Float64,2}, u::Array{Float64,2}, 
								F::Array{Float64,2}) # Roe solver

	L = zeros(Float64, N,7,7) # left hand eigenvectors
	R = zeros(Float64, N,7,7) # right hand eigenvectors

	for i=2:N-1 # main loop !
		
		drho = abs(u[i,1] - u[i+1,1]) # to cell boundary

		if drho <= 1e-12
			rho = 0.5 * (u[i,1] + u[i+1,1])
		else 
			rho = abs(gam0*drho/(u[i+1,1]^gam0 - u[i,1]^gam0))^(1/gamma)
		end

		vx = 0.5 * (u[i,2] + u[i+1,2])
		vy = 0.5 * (u[i,3] + u[i+1,3])
		vz = 0.5 * (u[i,4] + u[i+1,4])

		Bx = 0.5 * (u[i,5] + u[i+1,5])
		By = 0.5 * (u[i,6] + u[i+1,6])
		Bz = 0.5 * (u[i,7] + u[i+1,7])

		pg = 0.5 * (u[i,8] + u[i+1,8])
		S = pg/rho^(gamma-1) 

		v2 = vx^2 + vy^2 + vz^2
		B2 = Bx^2 + By^2 + Bz^2

		cs2 = max(0, gamma*abs(pg/rho))
		cs = sqrt(cs2)
		bbn2 = B2/rho
		bnx2 = Bx^2/rho

		root = max(0, (bbn2+cs2)^2 - 4*bnx2*cs2)
		root = sqrt(root)

		lf = sqrt(max(0, (bbn2+cs2+root)/2)) 	# fast mode
		la = sqrt(max(0, bnx2))					# alven mode
		ls = sqrt(max(0, (bbn2+cs2-root)/2))		# slow mode

		# resolve degeneracies

		Bt2 = By^2 + Bz^2
		sgnBx = sign(Bx) # -1, 0, 1

		bty = 1/sqrt(2)
		btz = 1/sqrt(2)

		if Bt2 > 1e-30
			bty = By/sqrt(Bt2)
			btz = Bz/sqrt(Bt2)
		end

		af = 1
		as = 1

		dl2 = lf^2 - ls^2

		if dl2 >= 1e-30
			af = sqrt(max(0,cs2 - ls^2))/sqrt(dl2)
			as = sqrt(max(0,lf^2 - cs2))/sqrt(dl2)
		end

		# eigenvectors

		sqrt_rho = sqrt(rho)

		L[i,1,1] =  af*(gamS*cs2+lf*vx) - as*ls*(bty*vy+btz*vz)*sgnBx
        L[i,1,2] = -af*lf
        L[i,1,3] =  as*ls*bty*sgnBx
        L[i,1,4] =  as*ls*btz*sgnBx
        L[i,1,5] =  cs*as*bty*sqrt_rho
        L[i,1,6] =  cs*as*btz*sqrt_rho
        L[i,1,7] =  af*cs2*rho/(gamma*S)
            
        L[i,2,1] =  btz*vy-bty*vz
        L[i,2,2] =  0
        L[i,2,3] = -btz
        L[i,2,4] =  bty
        L[i,2,5] = -btz*sgnBx*sqrt_rho
        L[i,2,6] =  bty*sgnBx*sqrt_rho
        L[i,2,7] =  0
            
        L[i,3,1] =  as*(gamS*cs2+ls*vx) + af*lf*(bty*vy+btz*vz)*sgnBx
        L[i,3,2] = -as*ls
        L[i,3,3] = -af*lf*bty*sgnBx
        L[i,3,4] = -af*lf*btz*sgnBx
        L[i,3,5] = -cs*af*bty*sqrt_rho
        L[i,3,6] = -cs*af*btz*sqrt_rho
        L[i,3,7] =  as*cs2*rho/(gamma*S)
            
        L[i,4,1] =  1/gamma
        L[i,4,2] =  0
        L[i,4,3] =  0
        L[i,4,4] =  0
        L[i,4,5] =  0
        L[i,4,6] =  0
        L[i,4,7] = -rho/(gamma*S)
            
        L[i,5,1] =  as*(gamS*cs2-ls*vx) - af*lf*(bty*vy+btz*vz)*sgnBx
        L[i,5,2] =  as*ls
        L[i,5,3] =  af*lf*bty*sgnBx
        L[i,5,4] =  af*lf*btz*sgnBx
        L[i,5,5] = -cs*af*bty*sqrt_rho
        L[i,5,6] = -cs*af*btz*sqrt_rho
        L[i,5,7] =  as*cs2*rho/(gamma*S)
            
        L[i,6,1] =  btz*vy-bty*vz
        L[i,6,2] =  0
        L[i,6,3] = -btz
        L[i,6,4] =  bty
        L[i,6,5] =  btz*sgnBx*sqrt_rho
        L[i,6,6] = -bty*sgnBx*sqrt_rho
        L[i,6,7] =  0
            
        L[i,7,1] =  af*(gamS*cs2-lf*vx) + as*ls*(bty*vy+btz*vz)*sgnBx
        L[i,7,2] =  af*lf
        L[i,7,3] = -as*ls*bty*sgnBx
        L[i,7,4] = -as*ls*btz*sgnBx
        L[i,7,5] =  cs*as*bty*sqrt_rho
        L[i,7,6] =  cs*as*btz*sqrt_rho
        L[i,7,7] =  af*cs2*rho/(gamma*S)

        for m=1:7
            L[i,1,m] *= 0.5/cs2
            L[i,2,m] *= 0.5
            L[i,3,m] *= 0.5/cs2 # 4 missing ?
            L[i,5,m] *= 0.5/cs2
            L[i,6,m] *= 0.5
        	L[i,7,m] *= 0.5/cs2
        end

		R[i,1,1] = af
        R[i,2,1] = af*(vx-lf)
        R[i,3,1] = af*vy + as*ls*bty*sgnBx
        R[i,4,1] = af*vz + as*ls*btz*sgnBx
        R[i,5,1] = cs*as*bty/sqrt_rho
        R[i,6,1] = cs*as*btz/sqrt_rho
        R[i,7,1] = af*S/rho
            
        R[i,1,2] =  0
        R[i,2,2] =  0
        R[i,3,2] = -btz
        R[i,4,2] =  bty
        R[i,5,2] = -btz*sgnBx/sqrt_rho
        R[i,6,2] =  bty*sgnBx/sqrt_rho
        R[i,7,2] =  0
            
        R[i,1,3] =  as
        R[i,2,3] =  as*(vx-ls)
        R[i,3,3] =  as*vy - af*lf*bty*sgnBx
        R[i,4,3] =  as*vz - af*lf*btz*sgnBx
        R[i,5,3] = -cs*af*bty/sqrt_rho
        R[i,6,3] = -cs*af*btz/sqrt_rho
        R[i,7,3] =  as*S/rho
            
        R[i,1,4] =  1
        R[i,2,4] =  vx
        R[i,3,4] =  vy
        R[i,4,4] =  vz
        R[i,5,4] =  0
        R[i,6,4] =  0
        R[i,7,4] =  gam0*S/rho
            
        R[i,1,5] =  as
        R[i,2,5] =  as*(vx+ls)
        R[i,3,5] =  as*vy + af*lf*bty*sgnBx
        R[i,4,5] =  as*vz + af*lf*btz*sgnBx
        R[i,5,5] = -cs*af*bty/sqrt_rho
        R[i,6,5] = -cs*af*btz/sqrt_rho
        R[i,7,5] =  as*S/rho
            
        R[i,1,6] =  0
        R[i,2,6] =  0
        R[i,3,6] = -btz
        R[i,4,6] =  bty
        R[i,5,6] =  btz*sgnBx/sqrt_rho
        R[i,6,6] = -bty*sgnBx/sqrt_rho
        R[i,7,6] =  0
            
        R[i,1,7] = af
        R[i,2,7] = af*(vx+lf)
        R[i,3,7] = af*vy - as*ls*bty*sgnBx
        R[i,4,7] = af*vz - as*ls*btz*sgnBx
        R[i,5,7] = cs*as*bty/sqrt_rho
        R[i,6,7] = cs*as*btz/sqrt_rho
        R[i,7,7] = af*S/rho

		sgnBt = sign(Bz) # enforce continuity

		if By != 0
			sgnBt = sign(Bx)
		end

		if cs >= la
			for m=1:7
				L[i,3,m] *= sgnBt 
				L[i,5,m] *= sgnBt 
				R[i,m,3] *= sgnBt 
				R[i,m,5] *= sgnBt 
			end
		else
			for m=1:7
				L[i,1,m] *= sgnBt 
				L[i,7,m] *= sgnBt 
				R[i,m,1] *= sgnBt 
				R[i,m,7] *= sgnBt 
			end
		end
	end  # for i

	return L, R
end # compute_eigenvectors_S

function compute_fluxes_S(q::Array{Float64,2}, u::Array{Float64,2})
	
	F = zeros(Float64, N, 7)
	
	pt = u[:,8] + (u[:,5].^2+u[:,6].^2+u[:,7].^2)./2 # total pressure from S

	F[:,1] = u[:,1].*u[:,2] 					# rho * vx
	F[:,2] = q[:,2].*u[:,2] + pt - u[:,5].^2 	# mvx*vx + Ptot - Bx^2
	F[:,3] = q[:,3].*u[:,2] - u[:,5].*u[:,6] 	# mvy*vx - Bx*By
	F[:,4] = q[:,4].*u[:,2] - u[:,5].*u[:,7] 	# mvy*vx - Bx*Bz
	F[:,5] = u[:,6].*u[:,2] - u[:,5].*u[:,3] 	# By*vx - Bx*vy
	F[:,6] = u[:,7].*u[:,2] - u[:,5].*u[:,4] 	# Bz*vx - Bx*vz
	F[:,7] = q[:,8].*u[:,2]	               		# S*vx

	return F
end



function compute_primitive_variables_S(q::Array{Float64,2})

	u = copy(q)  # = [rho,Mx,My,Mz,Bx,By,Bz,S] 
	
	u[:,2] ./= u[:,1] # vx = Mx/rho
	u[:,3] ./= u[:,1] # vy
	u[:,4] ./= u[:,1] # vz

	v2 = u[:,2].^2 + u[:,3].^2 + u[:,4].^2
	B2 = u[:,5].^2 + u[:,6].^2 + u[:,7].^2

	u[:,8] .*= u[:,1].^(gamma-1)	# Pressure from S

	return u # = [rho,vx,vy,vz,Bx,By,Bz,P(S)]; J&W eq. 2.23
end

# EE - code

function weno5_flux_difference_E(q::Array{Float64,2})
	
   #println("q0[1]      : $(q[1,:])")
   #println("q0[2]      : $(q[2,:])")
   #println("q0[3]      : $(q[3,:])")
   #println("q0[4]      : $(q[4,:])")
   #println("q0[132]    : $(q[132,:])")
   #println("q0[133]    : $(q[133,:])")
   #println("q0[iMax]   : $(q[iMax,:])")
	
	u = compute_primitive_variables_E(q) # [rho,vx,vy,vz,Bx,By,Bz,P(S)]  
	

#   println("iMin u : $(u[iMin,:])")
#   println("132  u : $(u[132,:])")
#   println("133  u : $(u[133,:])")
#   println("iMax u : $(u[iMax,:])")

	a = compute_eigenvalues(u)

	#println("iMin a : $(a[iMin,:])")
	#println("132  a : $(a[132,:])")
	#println("133  a : $(a[133,:])")
	#println("iMax a : $(a[iMax,:])")


	F = compute_fluxes_E(q,u)
 
    #println("iMin F : $(F[iMin,:])")
	#println("132  F : $(F[132,:])")
	#println("133  F : $(F[133,:])")
    #println("iMax F : $(F[iMax,:])")

	L, R = compute_eigenvectors_E(q,u,F)
	
   #println("iMin L : $(L[iMin,:,1])")
   #println("132  L : $(L[132, :,1])")
   #println("133  L : $(L[133, :,1])")
   #println("iMax L : $(L[iMax,:,1])")

   #println("iMin R : $(R[iMin,:,1])")
   #println("132  R : $(R[132, :,1])")
   #println("133  R : $(R[133, :,1])")
   #println("iMax R : $(R[iMax,:,1])")

   dF = weno5_interpolation(q,a,F,L,R)

   #println("1    dF : $(dF[1,:])")
   #println("2    dF : $(dF[2,:])")
   #println("3    dF : $(dF[3,:])")
   #println("4    dF : $(dF[4,:])")
   #println("5    dF : $(dF[5,:])")
   #println("iMin dF : $(dF[iMin,:])")
   #println("132  dF : $(dF[132, :])")
   #println("133  dF : $(dF[133, :])")
   #println("iMax dF : $(dF[iMax,:])")

	Q = zeros(Float64, N, 8)

	#Threads.@threads
	for i=iMin:iMax
		Q[i,1] = dF[i,1] - dF[i-1,1]
		Q[i,2] = dF[i,2] - dF[i-1,2]
		Q[i,3] = dF[i,3] - dF[i-1,3]
		Q[i,4] = dF[i,4] - dF[i-1,4]
		Q[i,5] = 0
		Q[i,6] = dF[i,5] - dF[i-1,5]
		Q[i,7] = dF[i,6] - dF[i-1,6]
		Q[i,8] = dF[i,7] - dF[i-1,7]
	end

   #println("1    Q : $(Q[1,:])")
   #println("2    Q : $(Q[2,:])")
   #println("3    Q : $(Q[3,:])")
   #println("iMin Q : $(Q[iMin,:])")
   #println("128  Q : $(Q[128, :])")
   #println("129  Q : $(Q[129, :])")
   #println("130  Q : $(Q[130, :])")
   #println("131  Q : $(Q[131, :])")
   #println("132  Q : $(Q[132, :])")
   #println("133  Q : $(Q[133, :])")
   #println("134  Q : $(Q[134, :])")
   #println("iMax Q : $(Q[iMax,:])")

	return Q
end

function compute_eigenvectors_E(q::Array{Float64,2}, u::Array{Float64,2}, 
								F::Array{Float64,2}) # Roe solver

	L = zeros(Float64, N,7,7) # left hand eigenvectors
	R = zeros(Float64, N,7,7) # right hand eigenvectors

	#Threads.@threads
	for i=2:N-1 # main loop !
		
		rho = (u[i,1] + u[i+1,1]) / 2 # to cell boundary

		vx = 0.5 * (u[i,2] + u[i+1,2])
		vy = 0.5 * (u[i,3] + u[i+1,3])
		vz = 0.5 * (u[i,4] + u[i+1,4])

		Bx = 0.5 * (u[i,5] + u[i+1,5])
		By = 0.5 * (u[i,6] + u[i+1,6])
		Bz = 0.5 * (u[i,7] + u[i+1,7])

		E = 0.5 * (q[i,8] + q[i+1,8]) # q NOT u

		v2 = vx^2 + vy^2 + vz^2
		B2 = Bx^2 + By^2 + Bz^2

		pg = (gamma-1) * (E - 0.5 * (rho*v2 + B2) )

		cs2 = max(0, gamma*abs(pg/rho))
		cs = sqrt(cs2)
		bbn2 = B2/rho
		bnx2 = Bx^2/rho

		root = max(0, (bbn2+cs2)^2 - 4*bnx2*cs2)
		root = sqrt(root)

		lf = sqrt(max(0, (bbn2+cs2+root)/2)) 	# fast mode
		la = sqrt(max(0, bnx2))					# alven mode
		ls = sqrt(max(0, (bbn2+cs2-root)/2))	# slow mode

		# resolve degeneracies

		Bt2 = By^2 + Bz^2
		sgnBx = sign(Bx) # -1, 0, 1

		bty = 1/sqrt(2)
		btz = 1/sqrt(2)

		if Bt2 >= 1e-30
			bty = By/sqrt(Bt2)
			btz = Bz/sqrt(Bt2)
		end

		af = 1
		as = 1

		dl2 = lf^2 - ls^2

		if dl2 >= 1e-30
			af = sqrt(max(0,cs2 - ls^2))/sqrt(dl2)
			as = sqrt(max(0,lf^2 - cs2))/sqrt(dl2)
		end

		# eigenvectors E version

		sqrt_rho = sqrt(rho)

		L[i,1,1] = af*(gam1*v2 + lf*vx) - as*ls*(bty*vy + btz*vz)*sgnBx
        L[i,1,2] = af*(gam0*vx - lf)
        L[i,1,3] = gam0*af*vy + as*ls*bty*sgnBx
        L[i,1,4] = gam0*af*vz + as*ls*btz*sgnBx
        L[i,1,5] = gam0*af*By + cs*as*bty*sqrt_rho
        L[i,1,6] = gam0*af*Bz + cs*as*btz*sqrt_rho
        L[i,1,7] = -gam0*af
                                                                   
        L[i,2,1] = btz*vy - bty*vz
        L[i,2,2] = 0
        L[i,2,3] = -btz
        L[i,2,4] = bty
        L[i,2,5] = -btz*sgnBx*sqrt_rho
        L[i,2,6] = bty*sgnBx*sqrt_rho
        L[i,2,7] = 0
                                                                   
        L[i,3,1] = as*(gam1*v2 + ls*vx) + af*lf*(bty*vy + btz*vz)*sgnBx
        L[i,3,2] = gam0*as*vx - as*ls
        L[i,3,3] = gam0*as*vy - af*lf*bty*sgnBx
        L[i,3,4] = gam0*as*vz - af*lf*btz*sgnBx
        L[i,3,5] = gam0*as*By - cs*af*bty*sqrt_rho 
        L[i,3,6] = gam0*as*Bz - cs*af*btz*sqrt_rho 
        L[i,3,7] = -gam0*as
                                                                   
        L[i,4,1] = -cs2/gam0 - 0.5*v2
        L[i,4,2] = vx
        L[i,4,3] = vy
        L[i,4,4] = vz
        L[i,4,5] = By
        L[i,4,6] = Bz
        L[i,4,7] = -1
                                                                   
        L[i,5,1] = as*(gam1*v2 - ls*vx) - af*lf*(bty*vy + btz*vz)*sgnBx
        L[i,5,2] = as*(gam0*vx+ls)
        L[i,5,3] = gam0*as*vy + af*lf*bty*sgnBx
        L[i,5,4] = gam0*as*vz + af*lf*btz*sgnBx
        L[i,5,5] = gam0*as*By - cs*af*bty*sqrt_rho 
        L[i,5,6] = gam0*as*Bz - cs*af*btz*sqrt_rho 
        L[i,5,7] = -gam0*as
                                                                   
        L[i,6,1] = btz*vy - bty*vz
        L[i,6,2] = 0
        L[i,6,3] = -btz
        L[i,6,4] = bty
        L[i,6,5] = btz*sgnBx*sqrt_rho
        L[i,6,6] = -bty*sgnBx*sqrt_rho
        L[i,6,7] = 0
                                                                   
        L[i,7,1] = af*(gam1*v2 - lf*vx) + as*ls*(bty*vy + btz*vz)*sgnBx
        L[i,7,2] = af*(gam0*vx + lf)
        L[i,7,3] = gam0*af*vy - as*ls*bty*sgnBx
        L[i,7,4] = gam0*af*vz - as*ls*btz*sgnBx
        L[i,7,5] = gam0*af*By + cs*as*bty*sqrt_rho
        L[i,7,6] = gam0*af*Bz + cs*as*btz*sqrt_rho
        L[i,7,7] = -gam0*af

        for m=1:7
            L[i,1,m] *= 0.5/cs2
            L[i,2,m] *= 0.5
            L[i,3,m] *= 0.5/cs2 
			L[i,4,m] *= -gam0/cs2
            L[i,5,m] *= 0.5/cs2
            L[i,6,m] *= 0.5
        	L[i,7,m] *= 0.5/cs2
        end

		R[i,1,1] = af
        R[i,2,1] = af*(vx - lf)
        R[i,3,1] = af*vy + as*ls*bty*sgnBx
        R[i,4,1] = af*vz + as*ls*btz*sgnBx
        R[i,5,1] = cs*as*bty/sqrt_rho
        R[i,6,1] = cs*as*btz/sqrt_rho
        R[i,7,1] = af*(lf^2 - lf*vx + 0.5*v2 - gam2*cs2) + as*ls*(bty*vy + btz*vz)*sgnBx 
           
        R[i,1,2] = 0
        R[i,2,2] = 0
        R[i,3,2] = -btz
        R[i,4,2] = bty
        R[i,5,2] = -btz*sgnBx/sqrt_rho
        R[i,6,2] = bty*sgnBx/sqrt_rho
        R[i,7,2] = bty*vz - btz*vy
                                                      
        R[i,1,3] = as
        R[i,2,3] = as*(vx-ls)
        R[i,3,3] = as*vy - af*lf*bty*sgnBx
        R[i,4,3] = as*vz - af*lf*btz*sgnBx
        R[i,5,3] = -cs*af*bty/sqrt_rho
        R[i,6,3] = -cs*af*btz/sqrt_rho
        R[i,7,3] = as*(ls^2 - ls*vx + 0.5*v2 - gam2*cs2) - af*lf*(bty*vy + btz*vz)*sgnBx
            
        R[i,1,4] = 1
        R[i,2,4] = vx
        R[i,3,4] = vy
        R[i,4,4] = vz
        R[i,5,4] = 0
        R[i,6,4] = 0
        R[i,7,4] = 0.5*v2
                                                       
        R[i,1,5] =  as
        R[i,2,5] =  as*(vx + ls)
        R[i,3,5] =  as*vy + af*lf*bty*sgnBx
        R[i,4,5] =  as*vz + af*lf*btz*sgnBx
        R[i,5,5] = -cs*af*bty/sqrt_rho
        R[i,6,5] = -cs*af*btz/sqrt_rho
        R[i,7,5] =  as*(ls^2 + ls*vx + 0.5*v2 - gam2*cs2) + af*lf*(bty*vy + btz*vz)*sgnBx
            
        R[i,1,6] =  0
        R[i,2,6] =  0
        R[i,3,6] = -btz
        R[i,4,6] =  bty
        R[i,5,6] =  btz*sgnBx/sqrt_rho
        R[i,6,6] = -bty*sgnBx/sqrt_rho
        R[i,7,6] =  bty*vz - btz*vy
                                                      
        R[i,1,7] = af
        R[i,2,7] = af*(vx + lf)
        R[i,3,7] = af*vy - as*ls*bty*sgnBx
        R[i,4,7] = af*vz - as*ls*btz*sgnBx
        R[i,5,7] = cs*as*bty/sqrt_rho
        R[i,6,7] = cs*as*btz/sqrt_rho
        R[i,7,7] = af*(lf^2 + lf*vx + 0.5*v2 - gam2*cs2) - as*ls*(bty*vy + btz*vz)*sgnBx

		sgnBt = sign(Bz) # enforce continuity

		if By != 0
			sgnBt = sign(Bx)
		end

		if cs >= la
			for m=1:7
				L[i,3,m] *= sgnBt 
				L[i,5,m] *= sgnBt 
				R[i,m,3] *= sgnBt 
				R[i,m,5] *= sgnBt 
			end
		else
			for m=1:7
				L[i,1,m] *= sgnBt 
				L[i,7,m] *= sgnBt 
				R[i,m,1] *= sgnBt 
				R[i,m,7] *= sgnBt 
			end
		end
	end  # for i

	return L, R
end # compute_eigenvectors_E

function compute_fluxes_E(q::Array{Float64,2}, u::Array{Float64,2})
	
	F = compute_fluxes_S(q,u) # all but 2 & 7 are the same !
	
	pt = u[:,8] + (u[:,5].^2+u[:,6].^2+u[:,7].^2)/2 # total pressure from E

	vB = u[:,2].*u[:,5] + u[:,3].*u[:,6] + u[:,4].*u[:,7] # v*B

	# replace only components 2 & 7

	F[:,2] = q[:,2].*u[:,2] + pt - u[:,5].^2	# mvx*vx + Ptot - Bx^2
	F[:,7] = (q[:,8]+pt) .* u[:,2] - u[:,5].*vB	# (E+Ptot)*vx - Bx * v*B

	return F
end

function compute_primitive_variables_E(q::Array{Float64,2})

	u = copy(q)  	# = [rho,Mx,My,Mz,Bx,By,Bz,E]
	
	u[:,2] ./= u[:,1] # vx = Mx/rho
	u[:,3] ./= u[:,1] # vy
	u[:,4] ./= u[:,1] # vz

	v2 = u[:,2].^2 + u[:,3].^2 + u[:,4].^2
	B2 = u[:,5].^2 + u[:,6].^2 + u[:,7].^2

	u[:,8] -= 0.5 * (u[:,1].*v2 + B2) # Pressure from E
	u[:,8] .*= (gamma-1)

	return u # = [rho,vx,vy,vz,Bx,By,Bz,P(E)] ; B&W eq. 2.23
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

function boundaries!(q::Array{Float64,2})

	for m=1:9
			
		for i=1:iMin-1
			q[i,m] = q[iMin,m]
		end

		for i=iMax+1:N
			q[i,m] = q[iMax,m]
		end
	end # m

	return q
end

function initial_conditions() # RJ95 2a

	sqrt4p = sqrt(4.0*pi)
	
	# conserved variables left & right
	# states:             rho   vx    vy   vz   Bx        By           Bz      Press
	ql = state2conserved(1.08, 1.2, 0.01, 0.5, 2/sqrt4p, 3.6/sqrt4p, 2/sqrt4p, 0.95)
	qr = state2conserved(1.0,    0,    0,   0, 2/sqrt4p, 4/sqrt4p,   2/sqrt4p, 1   )

	q0 = zeros(Float64, N, 9)

	for i = 1:N 

		if i < N/2+2
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

	E = P / (gamma-1) + 0.5 * (rho*v2+B2)
	S = P / rho^(gamma-1)

	return [rho, rho*vx, rho*vy, rho*vz, Bx, By, Bz, S, E]
end

function safe_sqrt(x::Array{Float64,1})
	
	bad = find(x .< 0)
	x[bad] = 0

	nBad = size(bad,1)

	if nBad > 0
		println("nBad = $nBad ")
	end

	return sqrt(x)
end

#end # module
