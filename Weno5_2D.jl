#module Weno5_2D

using LaTeXStrings

const tmax = 0.1		# end of time integration

const NBnd = 3			# number of boundary cells

const Nx = 256 + 2*NBnd	# number of cells including boundaries
const Ny = 64  + 2*NBnd	# number of cells including boundaries

const iMin = 1 + NBnd	# grid loop x minimum without boundaries
const iMax = Nx - NBnd	# grid loop x maximum without boundaries
const jMin = 1 + NBnd	# grid loop y minimum without boundaries
const jMax = Ny - NBnd	# grid loop y maximum without boundaries

const xsize = 4
const ysize = 1

const dx = xsize/(Nx - 2*NBnd)
const dy = ysize/(Ny - 2*NBnd)

const gam = 5.0/3.0	# adiabatic index & friends
const gam0 = 1-gam
const gam1 = (gam-1)/2
const gam2 = (gam-2)/(gam-1)
const gamS = (gam-1)/gam

const courFac = 0.8		# Courant Factor

const eps = 1e-6 		# Jiang & Wu eq 2.3++ 

export WENO5_2D

function WENO5_2D()

    q = initial_conditions() #  q=[rho,vx,vy,vz,Bx,By,Bz,S,E]

    bxb, byb = face_centered_bfld(q)

    boundaries!(q)

    t = 0
    nstep = 0

    while t < tmax

        dt, vxmax, vymax = timestep(q)

	    @printf "%d : t = %g, dt = %g, vmax = %g, %g \n" nstep t dt vxmax vymax
    
        q, bxb, byb = classical_RK4_step(q, bxb, byb, dt)
        
		t += dt

		nstep += 1

    end # while

	return

end # WENO5_2D

function classical_RK4_step(q0::Array{Float64,3}, bxb0::Array{Float64,2}, byb0::Array{Float64,2}, 
                            dt::Float64)

    fsy = zeros(Nx, Ny)
    gsx = zeros(Nx, Ny)

    # step 1

    q0_S = copy(q0[:,:,[1,2,3,4,5,6,7,8]])  # 8th component is S here
    dF0_S, fsy_S, tmp = weno5_flux_difference_S(q0_S)
    q0_S = q0_S - 1/2 * dt/dx * dF0_S        

    printq(q0_S,bxb0,byb0,fsy_S,tmp)

    rotate!(q0_S)
    dF0_S, tmp, gsx_S = weno5_flux_difference_S(q0_S)
    
    rotate!(q0_S)                           # rotating gsx not required
    rotate!(q0_S)
    
    q1_S = q0_S - 1/2 * dt/dy * dF0_S 

    q0_E = copy(q0[:,:,[1,2,3,4,5,6,7,9]])  # 8th component is S here
    dF0_E, fsy_E, tmp = weno5_flux_difference_S(q0_E)
    q0_E = q0_E - 1/2 * dt/dx * dF0_E        
    rotate!(q0_E)
    dF0_E, tmp, gsx_E = weno5_flux_difference_S(q0_E)
    rotate!(q0_E)                           # rotating gsx not required
    rotate!(q0_E)
    q1_E = q0_S - 1/2 * dt/dy * dF0_E

    #q1, fsy, gsx, idx = ES_Switch(q1_E, q1_S, fsy_S, gsx_S, fsy_E, gsx_E) 

    #bxb1 = copy(bxb0)
    #byb1 = copy(byb0)
    #fluxCT!(q1, bxb1, byb1, fsy, gsx)

    # step 2
    # step 3
    # step 4



    #bxb4 = (-bxb0 + bxb1 + 2*bxb2 + bxb3)/3
    #byb4 = (-byb0 + byb1 + 2*byb2 + byb3)/3

    return q4, bxb4, byb4

end # classical_RK4_step

function rotate!(q::Array{Float64,3})

    p = copy(q)

    for j = 1:size(q,2)
       @inbounds @simd for i = 1:size(q,1)

            q[i,j,2] = p[i,j,3] # vx -> vz
            q[i,j,3] = p[i,j,4] # vy -> vx
            q[i,j,4] = p[i,j,2] # vz -> vy
            q[i,j,5] = p[i,j,6] # Bx -> Bz
            q[i,j,6] = p[i,j,7] # By -> Bx
            q[i,j,7] = p[i,j,5] # Bz -> By
        end
    end

    return
end # rotate!

function fluxCT!(q::Array{Float64,3}, bxb::Array{Float64,2}, byb::Array{Float64,2}, 
                fsy::Array{Float64,2}, gsx::Array{Float64,2}, dt::Float64)


    fsy[1,:] = fsy[NBnd,:]
    fsy[Nx-NBnd+1,:] = fsy[Nx-NBnd,:]

    gsx[:,1] = gsx[:, NBnd]
    gsx[:,Nx-NBnd+1] = gsx[:, Nx-NBnd]

    Ox = zeros(Nx, Ny)

    for j = 1:Ny-1
        @inbounds @simd for i = 1:Nx-1
            Ox[i,j] = 0.5 * (gsx[i,j] + gsx[i+1,j] - fsy[i,j] - fsy[i,j+1])
        end
    end

    q1 = copy(q0)

    for j = 1:Ny-1
        @inbounds @simd for i = 1:Nx-1
            bxb[i,j] -= dt/dy * (Ox[i,j] - Ox[i,j-1])
            byb[i,j] -= dt/dx * (Ox[i,j] - Ox[i-1,j])
        end
    end

    return
end # fluxCT

# Entropy (S) code

function weno5_flux_difference_S(q_2D::Array{Float64,3})

    dq = SharedArray{Float64}(Nx, Ny, 8) * 0
    bsy = SharedArray{Float64}(Nx, Ny) * 0
    bsz = SharedArray{Float64}(Nx, Ny) * 0

   @sync @parallel for j = 1:Ny  # 1D along x for fixed y

        q = q_2D[:,j,:]

        u = compute_primitive_variables_S(q) # [rho,vx,vy,vz,Bx,By,Bz,P(S)]  
	    
        a = compute_eigenvalues(u)

	    F = compute_fluxes_S(q,u)

	    L, R = compute_eigenvectors_S(q,u,F)
	
	    dF = weno5_interpolation(q,a,F,L,R)

       @inbounds @simd for i = iMin:iMax
    		dq[i,j,1] = dF[i,1] - dF[i-1,1]
    		dq[i,j,2] = dF[i,2] - dF[i-1,2]
    		dq[i,j,3] = dF[i,3] - dF[i-1,3]
    		dq[i,j,4] = dF[i,4] - dF[i-1,4]
    		dq[i,j,5] = 0                   # fluxct scheme later
    		dq[i,j,6] = dF[i,5] - dF[i-1,5]
    		dq[i,j,7] = dF[i,6] - dF[i-1,6]
	    	dq[i,j,8] = dF[i,7] - dF[i-1,7]
	    end

        @inbounds @simd for i = iMin:iMax
            bsy[i,j] = dF[i,5] + 0.5 *(u[i,5]*u[i,3] + u[i+1,5]*u[i+1,3])
            bsz[i,j] = dF[i,6] + 0.5 *(u[i,5]*u[i,4] + u[i+1,5]*u[i+1,4])
        end
    end

    return dq, bsy, bsz
end

function compute_eigenvectors_S(q::Array{Float64,2}, u::Array{Float64,2}, 
								F::Array{Float64,2}) # Roe solver
    N = size(q, 1)

	L = zeros(Float64, N,7,7) # left hand eigenvectors
	R = zeros(Float64, N,7,7) # right hand eigenvectors

	for i=2:N-1
		
		drho = abs(u[i,1] - u[i+1,1]) # to cell boundary

		if drho <= 1e-12
			rho = 0.5 * (u[i,1] + u[i+1,1])
		else 
			rho = abs(gam0*drho/(u[i+1,1]^gam0 - u[i,1]^gam0))^(1/gam)
		end

		vx = 0.5 * (u[i,2] + u[i+1,2])
		vy = 0.5 * (u[i,3] + u[i+1,3])
		vz = 0.5 * (u[i,4] + u[i+1,4])

		Bx = 0.5 * (u[i,5] + u[i+1,5])
		By = 0.5 * (u[i,6] + u[i+1,6])
		Bz = 0.5 * (u[i,7] + u[i+1,7])

		pg = 0.5 * (u[i,8] + u[i+1,8])
		S = pg/rho^(gam-1) 

		v2 = vx^2 + vy^2 + vz^2
		B2 = Bx^2 + By^2 + Bz^2

		cs2 = max(0, gam*abs(pg/rho))
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
        L[i,1,7] =  af*cs2*rho/(gam*S)
            
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
        L[i,3,7] =  as*cs2*rho/(gam*S)
            
        L[i,4,1] =  1/gam
        L[i,4,2] =  0
        L[i,4,3] =  0
        L[i,4,4] =  0
        L[i,4,5] =  0
        L[i,4,6] =  0
        L[i,4,7] = -rho/(gam*S)
            
        L[i,5,1] =  as*(gamS*cs2-ls*vx) - af*lf*(bty*vy+btz*vz)*sgnBx
        L[i,5,2] =  as*ls
        L[i,5,3] =  af*lf*bty*sgnBx
        L[i,5,4] =  af*lf*btz*sgnBx
        L[i,5,5] = -cs*af*bty*sqrt_rho
        L[i,5,6] = -cs*af*btz*sqrt_rho
        L[i,5,7] =  as*cs2*rho/(gam*S)
            
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
        L[i,7,7] =  af*cs2*rho/(gam*S)

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
	
    const N = size(u,1)
    pt = zeros(Float64, N)

    @inbounds @simd for i = 1:N # total pressure from S
        pt[i] = u[i,8] + 0.5*(u[i,5].^2+u[i,6].^2+u[i,7].^2) 
    end

	F = zeros(Float64, N, 7)

    @inbounds @simd for i = 1:N
       F[i,1] = u[i,1]*u[i,2] 					    # rho * vx
       F[i,2] = q[i,2]*u[i,2] + pt[i] - u[i,5]^2 	# mvx*vx + Ptot - Bx^2
       F[i,3] = q[i,3]*u[i,2] - u[i,5]*u[i,6] 	    # mvy*vx - Bx*By
       F[i,4] = q[i,4]*u[i,2] - u[i,5]*u[i,7] 	    # mvy*vx - Bx*Bz
       F[i,5] = u[i,6]*u[i,2] - u[i,5]*u[i,3] 	    # By*vx - Bx*vy
       F[i,6] = u[i,7]*u[i,2] - u[i,5]*u[i,4]   	# Bz*vx - Bx*vz
       F[i,7] = q[i,8]*u[i,2]	               		# S*vx
    end

	return F # cell centered fluxes
end

function compute_primitive_variables_S(q::Array{Float64,2})

	u = copy(q)  # = [rho,Mx,My,Mz,Bx,By,Bz,S]

    @inbounds @simd for i = 1:size(u,1)
        u[i,2] /= u[i,1] # vx = Mx/rho
        u[i,3] /= u[i,1] # vy
        u[i,4] /= u[i,1] # vz

        u[i,8] *= u[i,1]^(gam-1)	# Pressure from S
    end

	return u # = [rho,vx,vy,vz,Bx,By,Bz,P(S)]; J&W eq. 2.23
end

# Energy (E) - code

function weno5_state_difference_E(q_2D::Array{Float64,3}, bxb::Array{Float64,2}, 
                                  byb::Array{Float64,2},  fsy::Array{Float64,2}, 
                                  gsx::Array{Float64,2},  dt::Float64)

    return 
end # weno5_state_difference_E

function compute_eigenvalues(u::Array{Float64,2})

    N = size(u, 1)

	B2  = u[:,5].^2 + u[:,6].^2 + u[:,7].^2
	vdB = u[:,2].*u[:,5] + u[:,3].*u[:,6] + u[:,4].*u[:,7]

	cs2 = gam * abs.(u[:,8]./u[:,1])
	bad = find(cs2 .< 0)
	cs2[bad] = 0

	bbn2 = B2./u[:,1]
	bnx2 = u[:,5].^2./u[:,1]

	root = safe_sqrt((bbn2+cs2).^2 - 4*bnx2.*cs2)

	lf = safe_sqrt((bbn2+cs2+root)/2)
	la = safe_sqrt(bnx2)
	ls = safe_sqrt((bbn2+cs2-root)/2)

	a = zeros(Float64, N, 7) # eigenvalues cell center

    @inbounds @simd for i=1:N 
        a[i,1] = u[i,2] - lf[i] # u is rotated, integration is along 2 direction
    	a[i,2] = u[i,2] - la[i]
	    a[i,3] = u[i,2] - ls[i]
	    a[i,4] = u[i,2]
    	a[i,5] = u[i,2] + ls[i]
    	a[i,6] = u[i,2] + la[i]
    	a[i,7] = u[i,2] + lf[i]
    end

	return a #  J&W 2.27+1

end

function ES_Switch(q_E::Array{Float64,3}, q_S::Array{Float64,3}, fsy_S::Array{Float64,2}, 
                   gsx_S::Array{Float64,2}, fsy_E::Array{Float64,2}, gsx_E::Array{Float64,2})
	
	q_SE = copy(q_E)
	q_SE[:,8] = E2S(q_E) 		# convert q(E) to q(S(E))

	dS = abs(q_SE[:,8] - q_S[:,8])

	bad = find(dS .>= 1e-5)

	q = zeros(N, 9)
	q[:,1:8] = q_S				# use q(S) by default
	q[bad,1:8] = q_SE[bad,:]  	# replace q at the bad locations
	q[:,9] = S2E(q)

    fsy = copy(fsy_S)
    fsy[bad] = fsy_E[bad]

    gsx = copy(gsx_S)
    gsx[bad] = gsx_E[bad]

	return q, fsy, gsx, bad
end


function face_centered_bfld(q::Array{Float64,3})

    bxb = SharedArray{Float64}(Nx, Ny) * 0 # needed on 2:N-2 only
    byb = SharedArray{Float64}(Nx, Ny) * 0

    for j = 2:Ny-2 # fourth order interpolation
        @inbounds @simd for i = 2:Nx-2
           bxb[i,j] = (-q[i-1,j,5] + 9*q[i,j,5] + 9*q[i+1,j,5] - q[i+2,j,5]) /16
           byb[i,j] = (-q[i,j-1,6] + 9*q[i,j,6] + 9*q[i,j+1,6] - q[i,j+2,6]) /16
        end
    end

    return bxb, byb
end

function boundaries!(q::Array{Float64,3})

    for j = 1:Ny
        q[1,j,:] = q[NBnd+1,j,:]
        q[2,j,:] = q[NBnd+1,j,:]
        q[3,j,:] = q[NBnd+1,j,:]
        q[Nx,j,:] = q[Nx-NBnd,j,:]
        q[Nx-1,j,:] = q[Nx-NBnd,j,:]
        q[Nx-2,j,:] = q[Nx-NBnd,j,:]
    end
    
    @inbounds @simd for i = 1:Nx
        q[i, 1,:] = q[i,NBnd+1,:]
        q[i, 2,:] = q[i,NBnd+1,:]
        q[i, 3,:] = q[i,NBnd+1,:]
        q[i, Ny,:] = q[i,Ny-NBnd,:]
        q[i, Ny-1,:] = q[i,Ny-NBnd,:]
        q[i, Ny-2,:] = q[i,Ny-NBnd,:]
    end

    return
end

function ES_Switch(q_E::Array{Float64,3}, q_S::Array{Float64,3})
	
	q_SE = copy(q_E)
	q_SE[:,:,8] = E2S(q_E) 		# convert q(E) to q(S(E))

	dS = abs(q_SE[:,:,8] - q_S[:,:,8])

	bad = find(dS .>= 1e-5)

	q = zeros(Nx, 9)
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
	
	S = (E - 0.5*mom2./rho - B2/2)*(gam-1)./rho.^(gam-1)
	
	return S  # Ryu+ 1993, eq. 2.3 + 1
end

function S2E(qS::Array{Float64,3})
	
	S = qS[:,:,8]
	rho = qS[:,:,1]
	mom2 = qS[:,:,2].^2 + qS[:,:,3].^2 + qS[:,:,4].^2
	B2 = qS[:,:,5].^2 + qS[:,:,6].^2 + qS[:,:,7].^2

	E = rho.^(gam-1).*S/(gam-1) + B2/2 + 0.5.*mom2./rho

	return E # Ryu+ 1993, eq. 2.3+1
end

function weno5_interpolation(q::Array{Float64,2}, a::Array{Float64,2},
							 F::Array{Float64,2}, L::Array{Float64,3},
							 R::Array{Float64,3})
    N = size(q,1)

	dF = zeros(Float64, N, 7)

	for i = NBnd:N-NBnd
		
    	Fs = zeros(Float64, 7)
	    Fsk = zeros(Float64, 6)
    	qsk = zeros(Float64, 6)
	    dFsk = zeros(Float64, 5)
    	dqsk = zeros(Float64, 5)

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
	
		for m = 1:7
			
			dF[i,m] = Fs[1]*R[i,m,1] + Fs[2]*R[i,m,2] + # J&W eq. 2.17
                   	  Fs[3]*R[i,m,3] + Fs[4]*R[i,m,4] +
                      Fs[5]*R[i,m,5] + Fs[6]*R[i,m,6] +
                      Fs[7]*R[i,m,7]
		end

	end # i

	return dF
end # weno5_interpolation()
function timestep(q::Array{Float64,3})

    vxmax, vymax = compute_global_vmax(q)

    dtx = dx/vxmax
    dty = dy/vymax

    dt = courFac /(1/dtx + 1/dty)

    return dt, vxmax, vymax
end

function compute_global_vmax(q::Array{Float64,3})

    vxmax = SharedArray{Float64}(Nx, Ny)
    vymax = SharedArray{Float64}(Nx, Ny)

    @sync @parallel for j = jMin:jMax # Jiang & Wu after eq 2.24
		for i = iMin:iMax
			
			rho = (q[i,j,1] + q[i+1,j,1])/2
           
			vx = (q[i,j,2] + q[i+1,j,2])/2 / rho
			vy = (q[i,j,3] + q[i+1,j,3])/2 / rho
			vz = (q[i,j,4] + q[i+1,j,4])/2 / rho

			Bx = (q[i,j,5] + q[i+1,j,5])/2
			By = (q[i,j,6] + q[i+1,j,6])/2
			Bz = (q[i,j,7] + q[i+1,j,7])/2

			S = (q[i,j,8] + q[i+1,j,8])/2

			pres = S * rho^(gam-1)

			cs2 = gam * abs(pres/rho)
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

    q = Array{Float64}(Nx, Ny, 9)*0

	# state variables left & right
	#        rho      vx    vy vz Bx        By      Bz    Press
	sl = [2.11820, 13.69836, 0, 0, 0, 19.33647, 19.33647, 65.8881]
	sr = [      1,        0, 0, 0, 0,  9.12871, 9.12871,      1.0]

	for j = 1:Ny
		for i = 1:Nx

			x = i * dx - NBnd*dx
			y = j * dy - NBnd*dy

			rr = sqrt((x - 1.5)^2 + (y - 0.5)^2)

			if x <= 1.2
                s = copy(sl)
			else 
                s = copy(sr)
			end

			if rr <= R0
				s[1] = 10
            end 
                
            if (rr > R0) && (rr <= R1) 
				s[1] = 1 + 9 * (R1-rr) / (R1-R0)
			end

			q[i,j,:] = state2conserved(s[1],s[2],s[3],s[4],s[5],s[6],s[7],s[8])
		end
	end

	return q
end

function state2conserved(rho::Float64, vx::Float64, vy::Float64, vz::Float64, 
                          Bx::Float64, By::Float64, Bz::Float64, P::Float64)
	
	v2 = vx^2 + vy^2 + vz^2
	B2 = Bx^2 + By^2 + Bz^2

	E = P / (gam-1) + 0.5 * (rho*v2+B2)
	S = P / rho^(gam-1)
	
    return [rho, rho*vx, rho*vy, rho*vz, Bx, By, Bz, S, E]
end

function safe_sqrt(x::Array{Float64,1})
	
	y = copy(x)
	bad = find(y .< 0)
	y[bad] = 0
		
	return sqrt.(y)
end

function printq(q::Array{Float64,3}, bxb::Array{Float64,2}, byb::Array{Float64,2},  
                fsy::Array{Float64,2}, gsx::Array{Float64,2}; ID="")

    xidx = [1,2,3,NBnd+1,Nx/2,Nx-NBnd,Nx/2-1,Nx/2,Nx/2+1,NBnd+1,Nx/2,Nx-NBnd]
    yidx = [1,2,3,NBnd+1,NBnd+1,NBnd+1,Ny/2,Ny/2,Ny/2,Ny-NBnd,Ny-NBnd,Ny-NBnd]

    @printf "%s \n" ID

    for i=1:length(xidx)
        
        x = Int64(xidx[i])
        y = Int64(yidx[i])

        @printf "%3i %3i " x-NBnd y-NBnd 
        @printf "%6.4g %6.4g %6.4g %6.4g " q[x,y,1] q[x,y,2] q[x,y,3] q[x,y,4] 
        @printf "%6.4g %6.4g %6.4g %6.4g   || " q[x,y,5] q[x,y,6] q[x,y,7] q[x,y,8] 
        @printf "%6.4g %6.4g %6.4g %6.4g\n" bxb[x,y] byb[x,y] fsy[x,y] gsx[x,y]
    end
    b+=1
    return
end

function printu(u::Array{Float64,2}; ID="")

    xidx = [NBnd+1,floor((Nx-2*NBnd)/4)+NBnd, Nx/2-1,Nx/2,Nx/2+1,floor(3*(Nx-2*NBnd)/4+NBnd),Nx-NBnd]

    @printf "%s \n" ID

    for i=1:length(xidx)
        
        x = Int64(xidx[i])

        @printf "%3i  " x-NBnd 
        @printf "%6.4g %6.4g %6.4g %6.4g " u[x,1] u[x,2] u[x,3] u[x,4] 
        @printf "%6.4g %6.4g %6.4g %6.4g \n" u[x,5] u[x,6] u[x,7] u[x,8] 
    end

    b+=1
    return
end

#end # module
