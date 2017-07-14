#module Weno5_2D

using Plots
using LaTeXStrings
using ArrayStatistics

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

const courFac = 0.8		# Courant Factor

const eps = 1e-6 		# Jiang & Wu eq 2.3++ 

export WENO5_2D

function WENO5_2D()

    q = initial_conditions() #  q=[rho,vx,vy,vz,Bx,By,Bz,S,E]

    bxb, byb = interpolateB_center2face(q)

    boundaries!(q, bxb, byb)

    t = 0
    nstep = 600

    cmap = ColMap(5)  # you could choose other maps of course

    img = Array{typeof(cmap[1])}(Ny,Nx,nstep,9)

     
    anim = @animate for i=1:nstep

        dt, vxmax, vymax = timestep(q)

	    @printf "\n%d : t = %g, dt = %g, vmax = %g, %g \n" i t dt vxmax vymax

        tic()
        q, bxb, byb = classical_RK4_step(q, bxb, byb, dt)
        toc()

        divb = compute_divB(bxb, byb)

        @printf "Max DivB = %g\n" maximum(abs.(divb))

		t += dt
        
        for i = 1:9
            @printf "   i = %d Maxq = %e Meanq = %e \n"  i  maximum(q[:,:,i]) mean(q[:,:,i]) 		
        end     

        #nstep += 1

        img = Arr2Img(q[:,:,1], cmap; range=[0.01,19]) # rho
        Plots.plot(img, layout=9,ylabel="rho",yticks=nothing, size=(2048,2048*0.3),colorbar=true)

        img= Arr2Img(q[:,:,2], cmap; range=[0, 340]) # vx
        Plots.plot!(img, subplot=2, ylabel="vx",yticks=nothing)

        img = Arr2Img(q[:,:,3], cmap; range=[-35, 35])
        Plots.plot!(img, subplot=3, ylabel="vy",yticks=nothing)
        
        img = Arr2Img(q[:,:,4], cmap; range=[-35, 35])
        Plots.plot!(img, subplot=4, ylabel="vz",yticks=nothing)

        img = Arr2Img(q[:,:,5], cmap; range=[1,10])
        Plots.plot!(img, subplot=5, ylabel="Bx",yticks=nothing)

        img = Arr2Img(q[:,:,6], cmap; range=[5,25])
        Plots.plot!(img, subplot=6, ylabel="By",yticks=nothing)
        
        img = Arr2Img(q[:,:,7], cmap; range=[5,25])
        Plots.plot!(img, subplot=7, ylabel="Bz",yticks=nothing)

        img = Arr2Img(q[:,:,8], cmap; range=[0,70])
        Plots.plot!(img, subplot=8, ylabel="S",yticks=nothing)

        img = Arr2Img(divb, cmap; range=[-1e-11,1e-11])
        Plots.plot!(img, subplot=9, ylabel="DivB",yticks=nothing)

        #Plots.savefig("/Users/jdonnert/Desktop/frames")

    end 

    mp4(anim, "/Users/jdonnert/Desktop/blob.mp4")
    gif(anim, "/Users/jdonnert/Desktop/blob.gif")

	return

end # WENO5_2D

"""
This is the classical Runge-Kutta Scheme, fourth order. 
Don't be scared, it's straight forward e.g. Shu 1988, 99.
In RK4, we compute flux differences from the previous state and apply the
corrected change to the original state. Finally all 4 intermediates states
are combined to give the full step. 
Because of the high order time integration, the scheme can be dimensionally 
unsplit, i.e. we compute flux differences separately for every dimension from
the same state and apply them together. This means the Roe solver / Weno functions
are inherently the same for 1D, 2D and 3D. As most of the time is spend here, these
are our primary optimizations targets. We rotate the state data to loop over y instead
of x, so we can use the same functions for all directions and vectorize.
"""
function classical_RK4_step(q0::Array{Float64,3}, bxb0::Array{Float64,2}, 
                            byb0::Array{Float64,2}, dt::Float64)

    fsy = Array{Float64}(Nx, Ny) * 0
    gsx = Array{Float64}(Nx, Ny) * 0

    # step 1
    q0_E = copy(q0[:,:,[1,2,3,4,5,6,7,9]])                          # 8th component is E here

    dFx0_E, fsy_E, tmp = weno5_flux_difference_E(q0_E)              # x-direction
    q0_E_rot = rotate_state(q0_E, "fwd")                            # rotate: x2z, y2x, z2y
    dFy0_E_rot, tmp, gsx_E_rot = weno5_flux_difference_E(q0_E_rot)  # y-direction
    dFy0_E = rotate_state(dFy0_E_rot, "bwd")                        # rotate back                  
    gsx_E = transpose(gsx_E_rot)
    q1_E = q0_E - 1/2 * dt/dx * dFx0_E  - 1/2 * dt/dy * dFy0_E      # rk4 step 1 for E state

    boundaries!(q1_E, fsy_E, gsx_E)
    Ox, Oxi, Oxj =  corner_fluxes(fsy_E, gsx_E)                     # includes shifted fluxes
    bxb1 = bxb0 - 1/2*dt/dy * (Ox - Oxj)                            # rk4 step of face Bfld 
    byb1 = byb0 - 1/2*dt/dx * (Oxi - Ox)
    q1_E[:,:,5:6] = interpolateB_face2center(bxb1, byb1)            # update cell centered bfld
    boundaries!(q1_E, fsy_E, gsx_E)                                 # complete E state

    q0_S = copy(q0[:,:,[1,2,3,4,5,6,7,8]])                          # 8th component is S here
    dFx0_S, fsy_S, tmp = weno5_flux_difference_S(q0_S)              # x-direction
    q0_S_rot = rotate_state(q0_S, "fwd")                            # x -> y
    dFy0_S_rot, tmp, gsx_S = weno5_flux_difference_S(q0_S_rot)      # y-direction
    dFy0_S = rotate_state(dFy0_S_rot,"bwd")                         # rotate back
    gsx_S = transpose(gsx_S)

    q1_S = q0_S - 1/2 * dt/dx * dFx0_S - 1/2 * dt/dy * dFy0_S       # rk4 step 1 for S state

    boundaries!(q1_S, fsy_S, gsx_S)
    Ox, Oxi, Oxj =  corner_fluxes(fsy_S, gsx_S)                     # includes shifted fluxes
    bxb1 = bxb0 - 1/2*dt/dy * (Ox - Oxj)
    byb1 = byb0 - 1/2*dt/dx * (Oxi - Ox)
    q1_S[:,:,5:6] = interpolateB_face2center(bxb1, byb1)
    boundaries!(q1_S, fsy_S, gsx_S)                                 # complete S state

    q1, fsy, gsx, idx = ES_Switch(q1_E, q1_S, fsy_S, gsx_S, fsy_E,  # select E or S state
                                  gsx_E) 
    boundaries!(q1, fsy, gsx)
    Ox, Oxi, Oxj =  corner_fluxes(fsy, gsx)                         # redo fluxct
    bxb1 = bxb0 - 1/2*dt/dy * (Ox - Oxj)
    byb1 = byb0 - 1/2*dt/dx * (Oxi - Ox)
    q1[:,:,5:6] = interpolateB_face2center(bxb1, byb1)
    boundaries!(q1, fsy, gsx)                                       # final state - 1st iteration

#printq(q1, bxb1, byb1, fsy,gsx, "FINAL 1")
    # step 2

    q1_E = copy(q1[:,:,[1,2,3,4,5,6,7,9]])                          
    dFx1_E, fsy_E, tmp = weno5_flux_difference_E(q1_E)
    q1_E_rot = rotate_state(q1_E, "fwd")                            
    dFy1_E_rot, tmp, gsx_E_rot = weno5_flux_difference_E(q1_E_rot)
    dFy1_E = rotate_state(dFy1_E_rot, "bwd")                                          
    gsx_E = transpose(gsx_E_rot)
    
    q2_E = q0_E - 1/2 * dt/dx * dFx1_E  - 1/2 * dt/dy * dFy1_E
 
    boundaries!(q2_E, fsy_E, gsx_E)
    Ox, Oxi, Oxj =  corner_fluxes(fsy_E, gsx_E)                     
    bxb2 = bxb0 - 1/2*dt/dy * (Ox - Oxj)                             
    byb2 = byb0 - 1/2*dt/dx * (Oxi - Ox)
    q2_E[:,:,5:6] = interpolateB_face2center(bxb2, byb2)
    boundaries!(q2_E, fsy_E, gsx_E)                                 

    q1_S = copy(q1[:,:,[1,2,3,4,5,6,7,8]])                          
    dFx1_S, fsy_S, tmp = weno5_flux_difference_S(q1_S)
    q1_S_rot = rotate_state(q1_S, "fwd")                            
    dFy1_S_rot, tmp, gsx_S = weno5_flux_difference_S(q1_S_rot)
    dFy1_S = rotate_state(dFy1_S_rot,"bwd")
    gsx_S = transpose(gsx_S)

    q2_S = q0_S - 1/2 * dt/dx * dFx1_S - 1/2 * dt/dy * dFy1_S 

    boundaries!(q2_S, fsy_S, gsx_S)
    Ox, Oxi, Oxj =  corner_fluxes(fsy_S, gsx_S)                     
    bxb2 = bxb0 - 1/2*dt/dy * (Ox - Oxj)
    byb2 = byb0 - 1/2*dt/dx * (Oxi - Ox)
    q2_S[:,:,5:6] = interpolateB_face2center(bxb2, byb2)
    boundaries!(q2_S, fsy_S, gsx_S)                                 

    q2, fsy, gsx, idx = ES_Switch(q2_E, q2_S, fsy_S, gsx_S, fsy_E, gsx_E)

    boundaries!(q2, fsy, gsx)
    Ox, Oxi, Oxj =  corner_fluxes(fsy, gsx)                         
    bxb2 = bxb0 - 1/2*dt/dy * (Ox - Oxj)
    byb2 = byb0 - 1/2*dt/dx * (Oxi - Ox)
    q2[:,:,5:6] = interpolateB_face2center(bxb2, byb2)
    boundaries!(q2, fsy, gsx)                                       

    # step 3

    q2_E = copy(q2[:,:,[1,2,3,4,5,6,7,9]])                          
    dFx2_E, fsy_E, tmp = weno5_flux_difference_E(q2_E)
    q2_E_rot = rotate_state(q2_E, "fwd")                            
    dFy2_E_rot, tmp, gsx_E_rot = weno5_flux_difference_E(q2_E_rot)
    dFy2_E = rotate_state(dFy2_E_rot, "bwd")                                          
    gsx_E = transpose(gsx_E_rot)
    
    q3_E = q0_E - dt/dx * dFx2_E  - dt/dy * dFy2_E
 
    boundaries!(q3_E, fsy_E, gsx_E)
    Ox, Oxi, Oxj =  corner_fluxes(fsy_E, gsx_E)                     
    bxb3 = bxb0 - dt/dy * (Ox - Oxj)                             
    byb3 = byb0 - dt/dx * (Oxi - Ox)
    q3_E[:,:,5:6] = interpolateB_face2center(bxb3, byb3)
    boundaries!(q3_E, fsy_E, gsx_E)                                 

    q2_S = copy(q2[:,:,[1,2,3,4,5,6,7,8]])                          
    dFx2_S, fsy_S, tmp = weno5_flux_difference_S(q2_S)
    q2_S_rot = rotate_state(q2_S, "fwd")                            
    dFy2_S_rot, tmp, gsx_S = weno5_flux_difference_S(q2_S_rot)
    dFy2_S = rotate_state(dFy2_S_rot,"bwd")
    gsx_S = transpose(gsx_S)

    q3_S = q0_S - dt/dx * dFx2_S - dt/dy * dFy2_S 

    boundaries!(q3_S, fsy_S, gsx_S)
    Ox, Oxi, Oxj =  corner_fluxes(fsy_S, gsx_S)                     
    bxb3 = bxb0 - dt/dy * (Ox - Oxj)
    byb3 = byb0 - dt/dx * (Oxi - Ox)
    q3_S[:,:,5:6] = interpolateB_face2center(bxb3, byb3)
    boundaries!(q3_S, fsy_S, gsx_S)                                 

    q3, fsy, gsx, idx = ES_Switch(q3_E, q3_S, fsy_S, gsx_S, fsy_E, gsx_E)

    boundaries!(q3, fsy, gsx)
    Ox, Oxi, Oxj =  corner_fluxes(fsy, gsx)                         
    bxb3 = bxb0 - dt/dy * (Ox - Oxj)
    byb3 = byb0 - dt/dx * (Oxi - Ox)
    q3[:,:,5:6] = interpolateB_face2center(bxb3, byb3)
    boundaries!(q3, fsy, gsx)                                       

    # step 4

    q3_E = copy(q3[:,:,[1,2,3,4,5,6,7,9]])                          
    dFx3_E, fsy_E, tmp = weno5_flux_difference_E(q3_E)
    q3_E_rot = rotate_state(q3_E, "fwd")                            
    dFy3_E_rot, tmp, gsx_E_rot = weno5_flux_difference_E(q3_E_rot)
    dFy3_E = rotate_state(dFy3_E_rot, "bwd")                                          
    gsx_E = transpose(gsx_E_rot)
    
    q4_E = 1/3 * (-q0_E + q1_E + 2*q2_E + q3_E)
    q4_E += -1/6 * dt/dx * dFx3_E  - 1/6 * dt/dy * dFy3_E

    boundaries!(q4_E, fsy_E, gsx_E)
    Ox, Oxi, Oxj =  corner_fluxes(fsy_E, gsx_E)                     
    bxb4 = 1/3 * (-bxb0 + bxb1 + 2*bxb2 + bxb3) - 1/6 * dt/dy * (Ox - Oxj)
    byb4 = 1/3 * (-byb0 + byb1 + 2*byb2 + byb3) - 1/6 * dt/dx * (Oxi - Ox)
    q4_E[:,:,5:6] = interpolateB_face2center(bxb4, byb4)
    boundaries!(q4_E, fsy_E, gsx_E)       

    q3_S = copy(q3[:,:,[1,2,3,4,5,6,7,8]])                          
    dFx3_S, fsy_S, tmp = weno5_flux_difference_S(q3_S)
    q3_S_rot = rotate_state(q3_S, "fwd")                            
    dFy3_S_rot, tmp, gsx_S_rot = weno5_flux_difference_S(q3_S_rot)
    dFy3_S = rotate_state(dFy3_S_rot, "bwd")                                          
    gsx_S = transpose(gsx_S_rot)
    
    q4_S = 1/3 * (-q0_S + q1_S + 2*q2_S + q3_S)
    q4_S += -1/6 * dt/dx * dFx3_S  - 1/6 * dt/dy * dFy3_S

    boundaries!(q4_S, fsy_S, gsx_S)
    Ox, Oxi, Oxj =  corner_fluxes(fsy_S, gsx_S)                     
    bxb4 = 1/3 * (-bxb0 + bxb1 + 2*bxb2 + bxb3) - 1/6 * dt/dy * (Ox - Oxj)
    byb4 = 1/3 * (-byb0 + byb1 + 2*byb2 + byb3) - 1/6 * dt/dx * (Oxi - Ox)
    q4_S[:,:,5:6] = interpolateB_face2center(bxb4, byb4)
    boundaries!(q4_S, fsy_S, gsx_S)  

    q4, fsy, gsx, idx = ES_Switch(q4_E, q4_S, fsy_S, gsx_S, fsy_E, gsx_E)

    boundaries!(q4, fsy, gsx)
    Ox, Oxi, Oxj =  corner_fluxes(fsy, gsx)                         
    bxb4 = 1/3 * (-bxb0 + bxb1 + 2*bxb2 + bxb3) - 1/6 * dt/dy * (Ox - Oxj)
    byb4 = 1/3 * (-byb0 + byb1 + 2*byb2 + byb3) - 1/6 * dt/dx * (Oxi - Ox)
    q4[:,:,5:6] = interpolateB_face2center(bxb4, byb4)
    boundaries!(q4, bxb4, byb4)                                       

    printq(q4, bxb4, byb4, fsy, gsx, "Final4")
   
    return q4, bxb4, byb4

end # classical_RK4_step

function corner_fluxes(fsy::Array{Float64,2}, gsx::Array{Float64,2})

    const Nfx = size(fsy,1)
    const Nfy = size(fsy,2)

    Ox  = zeros(Float64, Nx, Ny)
    Oxi = zeros(Float64, Nx, Ny) # i shifted corner flux 
    Oxj = zeros(Float64, Nx, Ny) # j shifted corner flux 

    @inbounds for j = 2:Nfy-1
        @inbounds @simd for i = 2:Nfx-1
            Ox[i,j] = 0.5 * (gsx[i,j] + gsx[i+1,j] - fsy[i,j] - fsy[i,j+1])
        end
    end

    @inbounds for j = 2:Nfy-1
        @inbounds @simd for i = 2:Nfx-1
            Oxi[i,j] = Ox[i-1,j]
            Oxj[i,j] = Ox[i,j-1]
        end
    end

    return Ox, Oxi, Oxj
end

function interpolateB_face2center(bxb::Array{Float64,2}, byb::Array{Float64,2})

    Bxy = Array{Float64}(Nx,Ny,2)

    @inbounds for j = NBnd+1:Ny-2
        @inbounds @simd for i = NBnd+1:Nx-2 # 4th order
            Bxy[i,j,1] = (-bxb[i-2,j] + 9.0*bxb[i-1,j] + 9.0*bxb[i,j]- bxb[i+1,j])/16.0
            Bxy[i,j,2] = (-byb[i,j-2] + 9.0*byb[i,j-1] + 9.0*byb[i,j]- byb[i,j+1])/16.0
        end
    end

    return Bxy
end 

# Entropy (S) code


function weno5_flux_difference_S(q::Array{Float64,3})

    const Nqx = size(q,1)                       # q might be rotated
    const Nqy = size(q,2)
    
    u = compute_primitive_variables_S(q)        # [rho,vx,vy,vz,Bx,By,Bz,P(S)]  
    
    a = compute_eigenvalues(u)

    F = compute_fluxes_S(q,u)
    
    #L, R = compute_eigenvectors_S(q,u,F)
    L, R = compute_eigenvectors_S_vec(q,u,F,gam)

    dF = weno5_interpolation(q,a,F,L,R)

    dq = Array{Float64}(Nqx, Nqy, 8)
    bsy = Array{Float64}(Nqx, Nqy)
    bsz = Array{Float64}(Nqx, Nqy)

    @inbounds for j = NBnd:Nqy-NBnd+1           # 1D along x for fixed y
        @inbounds @simd for i = NBnd:Nqx-NBnd+1 # only data domain

    		dq[i,j,1] = dF[i,j,1] - dF[i-1,j,1]
    		dq[i,j,2] = dF[i,j,2] - dF[i-1,j,2]
    		dq[i,j,3] = dF[i,j,3] - dF[i-1,j,3]
    		dq[i,j,4] = dF[i,j,4] - dF[i-1,j,4]
    		dq[i,j,5] = 0
    		dq[i,j,6] = dF[i,j,5] - dF[i-1,j,5]
    		dq[i,j,7] = dF[i,j,6] - dF[i-1,j,6]
	    	dq[i,j,8] = dF[i,j,7] - dF[i-1,j,7]
	    end

        @inbounds @simd for i = NBnd:Nqx-NBnd+1
            bsy[i,j] = dF[i,j,5] + 0.5 *(u[i,j,5]*u[i,j,3] + u[i+1,j,5]*u[i+1,j,3])
            bsz[i,j] = dF[i,j,6] + 0.5 *(u[i,j,5]*u[i,j,4] + u[i+1,j,5]*u[i+1,j,4])
        end
    end

    return dq, bsy, bsz
end

function compute_eigenvectors_S(q::Array{Float64,3}, u::Array{Float64,3}, 
								F::Array{Float64,3}) # Roe solver
    const gam0 = 1-gam
    const gam1 = (gam-1)/2
    const gam2 = (gam-2)/(gam-1)
    const gamS = (gam-1)/gam

    Nux = size(u, 1)
    Nuy = size(u, 2)

	L = zeros(Float64, Nux, Nuy, 7,7) # left hand eigenvectors
    R = zeros(Float64, Nux, Nuy, 7,7) # right hand eigenvectors

	@inbounds for j = 1:Nuy-1
	    @inbounds for i = 1:Nux-1
		
		    drho = abs(u[i,j,1] - u[i+1,j,1]) # to cell boundary
    
    		if drho <= 1e-12
    			rho = 0.5 * (u[i,j,1] + u[i+1,j,1])
    		else 
    			rho = abs(gam0*drho/(u[i+1,j,1]^gam0 - u[i,j,1]^gam0))^(1/gam)
    		end

    		vx = 0.5 * (u[i,j,2] + u[i+1,j,2])
		    vy = 0.5 * (u[i,j,3] + u[i+1,j,3])
    		vz = 0.5 * (u[i,j,4] + u[i+1,j,4])

    		Bx = 0.5 * (u[i,j,5] + u[i+1,j,5])
    		By = 0.5 * (u[i,j,6] + u[i+1,j,6])
    		Bz = 0.5 * (u[i,j,7] + u[i+1,j,7])

    		pg = 0.5 * (u[i,j,8] + u[i+1,j,8])

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

            if sgnBx == 0
                sgnBx = 1
            end

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
    
    		L[i,j,1,1] =  af*(gamS*cs2+lf*vx) - as*ls*(bty*vy+btz*vz)*sgnBx
            L[i,j,1,2] = -af*lf
            L[i,j,1,3] =  as*ls*bty*sgnBx
            L[i,j,1,4] =  as*ls*btz*sgnBx
            L[i,j,1,5] =  cs*as*bty*sqrt_rho
            L[i,j,1,6] =  cs*as*btz*sqrt_rho
            L[i,j,1,7] =  af*cs2*rho/(gam*S)

            L[i,j,2,1] =  btz*vy-bty*vz
            L[i,j,2,2] =  0
            L[i,j,2,3] = -btz
            L[i,j,2,4] =  bty
            L[i,j,2,5] = -btz*sgnBx*sqrt_rho
            L[i,j,2,6] =  bty*sgnBx*sqrt_rho
            L[i,j,2,7] =  0
            
            L[i,j,3,1] =  as*(gamS*cs2+ls*vx) + af*lf*(bty*vy+btz*vz)*sgnBx
            L[i,j,3,2] = -as*ls
            L[i,j,3,3] = -af*lf*bty*sgnBx
            L[i,j,3,4] = -af*lf*btz*sgnBx
            L[i,j,3,5] = -cs*af*bty*sqrt_rho
            L[i,j,3,6] = -cs*af*btz*sqrt_rho
            L[i,j,3,7] =  as*cs2*rho/(gam*S)
            
            L[i,j,4,1] =  1/gam
            L[i,j,4,2] =  0
            L[i,j,4,3] =  0
            L[i,j,4,4] =  0
            L[i,j,4,5] =  0
            L[i,j,4,6] =  0
            L[i,j,4,7] = -rho/(gam*S)
            
            L[i,j,5,1] =  as*(gamS*cs2-ls*vx) - af*lf*(bty*vy+btz*vz)*sgnBx
            L[i,j,5,2] =  as*ls
            L[i,j,5,3] =  af*lf*bty*sgnBx
            L[i,j,5,4] =  af*lf*btz*sgnBx
            L[i,j,5,5] = -cs*af*bty*sqrt_rho
            L[i,j,5,6] = -cs*af*btz*sqrt_rho
            L[i,j,5,7] =  as*cs2*rho/(gam*S)
            
            L[i,j,6,1] =  btz*vy-bty*vz
            L[i,j,6,2] =  0
            L[i,j,6,3] = -btz
            L[i,j,6,4] =  bty
            L[i,j,6,5] =  btz*sgnBx*sqrt_rho
            L[i,j,6,6] = -bty*sgnBx*sqrt_rho
            L[i,j,6,7] =  0
            
            L[i,j,7,1] =  af*(gamS*cs2-lf*vx) + as*ls*(bty*vy+btz*vz)*sgnBx
            L[i,j,7,2] =  af*lf
            L[i,j,7,3] = -as*ls*bty*sgnBx
            L[i,j,7,4] = -as*ls*btz*sgnBx
            L[i,j,7,5] =  cs*as*bty*sqrt_rho
            L[i,j,7,6] =  cs*as*btz*sqrt_rho
            L[i,j,7,7] =  af*cs2*rho/(gam*S)

            for m=1:7
                L[i,j,1,m] *= 0.5/cs2
                L[i,j,2,m] *= 0.5
                L[i,j,3,m] *= 0.5/cs2 # 4 missing ?
                L[i,j,5,m] *= 0.5/cs2
                L[i,j,6,m] *= 0.5
            	L[i,j,7,m] *= 0.5/cs2
            end

	    	R[i,j,1,1] = af
            R[i,j,2,1] = af*(vx-lf)
            R[i,j,3,1] = af*vy + as*ls*bty*sgnBx
            R[i,j,4,1] = af*vz + as*ls*btz*sgnBx
            R[i,j,5,1] = cs*as*bty/sqrt_rho
            R[i,j,6,1] = cs*as*btz/sqrt_rho
            R[i,j,7,1] = af*S/rho
            
            R[i,j,1,2] =  0
            R[i,j,2,2] =  0
            R[i,j,3,2] = -btz
            R[i,j,4,2] =  bty
            R[i,j,5,2] = -btz*sgnBx/sqrt_rho
            R[i,j,6,2] =  bty*sgnBx/sqrt_rho
            R[i,j,7,2] =  0
            
            R[i,j,1,3] =  as
            R[i,j,2,3] =  as*(vx-ls)
            R[i,j,3,3] =  as*vy - af*lf*bty*sgnBx
            R[i,j,4,3] =  as*vz - af*lf*btz*sgnBx
            R[i,j,5,3] = -cs*af*bty/sqrt_rho
            R[i,j,6,3] = -cs*af*btz/sqrt_rho
            R[i,j,7,3] =  as*S/rho
            
            R[i,j,1,4] =  1
            R[i,j,2,4] =  vx
            R[i,j,3,4] =  vy
            R[i,j,4,4] =  vz
            R[i,j,5,4] =  0
            R[i,j,6,4] =  0
            R[i,j,7,4] =  gam0*S/rho
            
            R[i,j,1,5] =  as
            R[i,j,2,5] =  as*(vx+ls)
            R[i,j,3,5] =  as*vy + af*lf*bty*sgnBx
            R[i,j,4,5] =  as*vz + af*lf*btz*sgnBx
            R[i,j,5,5] = -cs*af*bty/sqrt_rho
            R[i,j,6,5] = -cs*af*btz/sqrt_rho
            R[i,j,7,5] =  as*S/rho
            
            R[i,j,1,6] =  0
            R[i,j,2,6] =  0
            R[i,j,3,6] = -btz
            R[i,j,4,6] =  bty
            R[i,j,5,6] =  btz*sgnBx/sqrt_rho
            R[i,j,6,6] = -bty*sgnBx/sqrt_rho
            R[i,j,7,6] =  0
            
            R[i,j,1,7] = af
            R[i,j,2,7] = af*(vx+lf)
            R[i,j,3,7] = af*vy - as*ls*bty*sgnBx
            R[i,j,4,7] = af*vz - as*ls*btz*sgnBx
            R[i,j,5,7] = cs*as*bty/sqrt_rho
            R[i,j,6,7] = cs*as*btz/sqrt_rho
            R[i,j,7,7] = af*S/rho

            # enforce continuity

		    if By != 0
			    sgnBt = sign(By)
            else
		        sgnBt = sign(Bz) 
		    end

            if sgnBt == 0
                sgnBt = 1
            end

    		if cs >= la
    			for m=1:7
    				L[i,j,3,m] *= sgnBt 
    				L[i,j,5,m] *= sgnBt 
    				R[i,j,m,3] *= sgnBt 
    				R[i,j,m,5] *= sgnBt 
    			end
    		else
    			for m=1:7
    				L[i,j,1,m] *= sgnBt 
    				L[i,j,7,m] *= sgnBt 
    				R[i,j,m,1] *= sgnBt 
    				R[i,j,m,7] *= sgnBt 
    			end
    		end
	    end  # for i
    end # for j

    return sdata(L), sdata(R)
end # compute_eigenvectors_S


function compute_eigenvectors_S_vec(q::Array{Float64,3}, u::Array{Float64,3}, 
								    F::Array{Float64,3}, gam::Float64) # Roe solver
    const gam0 = 1-gam
    const gam1 = (gam-1)/2
    const gam2 = (gam-2)/(gam-1)
    const gamS = (gam-1)/gam
    
    const Nqx = size(q,1)
    const Nqy = size(q,2)

    rho = zeros(Float64, Nqx, Nqy)
	
    # to cell boundary

    @inbounds for j = 1:Nqy-1
	    @inbounds for i = 1:Nqx-1

            drho = abs(u[i,j,1] - u[i+1,j,1]) 

            if drho <= 1e-12
    			rho[i,j] = 0.5 * (u[i,j,1] + u[i+1,j,1])
    		else 
    		    rho[i,j] = abs(gam0*drho/(u[i+1,j,1]^gam0 - u[i,j,1]^gam0))^(1/gam)
    		end
        end
    end

    vx = zeros(Float64, Nqx, Nqy)
    vy = zeros(Float64, Nqx, Nqy)
    vz = zeros(Float64, Nqx, Nqy)
    Bx = zeros(Float64, Nqx, Nqy)
    By = zeros(Float64, Nqx, Nqy)
    Bz = zeros(Float64, Nqx, Nqy)
    pg = zeros(Float64, Nqx, Nqy)
        
	@inbounds for j = 1:Nqy-1
	    @inbounds @simd for i = 1:Nqx-1
		
            vx[i,j] = 0.5 * (u[i,j,2] + u[i+1,j,2])
		    vy[i,j] = 0.5 * (u[i,j,3] + u[i+1,j,3])
    		vz[i,j] = 0.5 * (u[i,j,4] + u[i+1,j,4])

    		Bx[i,j] = 0.5 * (u[i,j,5] + u[i+1,j,5])
    		By[i,j] = 0.5 * (u[i,j,6] + u[i+1,j,6])
    		Bz[i,j] = 0.5 * (u[i,j,7] + u[i+1,j,7])

    		pg[i,j] = 0.5 * (u[i,j,8] + u[i+1,j,8])
        end
    end
    
    S = Array{Float64}( Nqx, Nqy)
    v2 = Array{Float64}( Nqx, Nqy)
    bbn2 = Array{Float64}( Nqx, Nqy)
    sqrt_rho = Array{Float64}(Nqx, Nqy)
	
    @inbounds for j = 1:Nqy
	    @inbounds @simd for i = 1:Nqx

            sqrt_rho[i,j] = sqrt(rho[i,j])
            S[i,j] = pg[i,j]/rho[i,j]^(gam-1) 

    		v2[i,j] = vx[i,j]^2 + vy[i,j]^2 + vz[i,j]^2

    		B2 = Bx[i,j]^2 + By[i,j]^2 + Bz[i,j]^2
    		bbn2[i,j] = B2/rho[i,j]
        end
    end

    cs2 = Array{Float64}(Nqx, Nqy)
    bnx2 = Array{Float64}(Nqx, Nqy)

    @. cs2 = max(0, gam*abs(pg/rho))
	@. bnx2 = Bx^2 / rho

    cs =  Array{Float64}(Nqx, Nqy)
    root =  Array{Float64}(Nqx, Nqy)
    lf =  Array{Float64}(Nqx, Nqy) # fast mode
    la =  Array{Float64}(Nqx, Nqy) # alven mode
    ls =  Array{Float64}(Nqx, Nqy) # slow mode
    Bt2 =  Array{Float64}(Nqx, Nqy)

    @. cs = sqrt(cs2)
    @. root = sqrt(max(0, (bbn2+cs2)^2 - 4*bnx2*cs2))
    @. lf = sqrt(max(0, (bbn2+cs2+root)/2)) 	
    @. la = sqrt(max(0, bnx2))					        
	@. ls = sqrt(max(0, (bbn2+cs2-root)/2))	
    @. Bt2 = By^2 + Bz^2

    # resolve degeneracies
	
    sgnBx =  Array{Float64}(Nqx, Nqy)
    @. sgnBx = sign(Bx) # -1, 0, 1
    sgnBx[find(sgnBx .== 0)] = 1

    bty =  Array{Float64}(Nqx, Nqy)
    btz =  Array{Float64}(Nqx, Nqy)
    @. bty = By/sqrt(Bt2)
    @. btz = Bz/sqrt(Bt2)
    bad = find(Bt2 .<= 1e-30)
    bty[bad] = 1/sqrt(2)
    btz[bad] = 1/sqrt(2)

    dl2 =  Array{Float64}(Nqx, Nqy)
    @. dl2 = lf^2 - ls^2

    af = fill(1.0, Nqx, Nqy)
    as = fill(1.0, Nqx, Nqy)

    @inbounds for j = 1:Nqy
	    @inbounds @simd for i = 1:Nqx-1

            if dl2[i,j] >= 1e-30
                af[i,j] = sqrt(max(0, cs2[i,j] - ls[i,j]^2)/dl2[i,j])
                as[i,j] = sqrt(max(0, lf[i,j]^2 - cs2[i,j])/dl2[i,j])
            end
        end
    end

    L = Array{Float64}( Nqx, Nqy, 7,7) # left hand eigenvectors
    R = Array{Float64}( Nqx, Nqy, 7,7) # right hand eigenvectors

    @inbounds for j = 1:Nqy-1 # eigenvectors S version
	    @inbounds @simd for i = 1:Nqx-1

    		L[i,j,1,1] =  af[i,j]*(gamS*cs2[i,j]+lf[i,j]*vx[i,j]) - 
                          as[i,j]*ls[i,j]*(bty[i,j]*vy[i,j]+btz[i,j]*vz[i,j])*sgnBx[i,j]
            L[i,j,1,2] = -af[i,j]*lf[i,j]
            L[i,j,1,3] =  as[i,j]*ls[i,j]*bty[i,j]*sgnBx[i,j]
            L[i,j,1,4] =  as[i,j]*ls[i,j]*btz[i,j]*sgnBx[i,j]
            L[i,j,1,5] =  cs[i,j]*as[i,j]*bty[i,j]*sqrt_rho[i,j]
            L[i,j,1,6] =  cs[i,j]*as[i,j]*btz[i,j]*sqrt_rho[i,j]
            L[i,j,1,7] =  af[i,j]*cs2[i,j]*rho[i,j]/(gam*S[i,j])
    
            L[i,j,2,1] =  btz[i,j]*vy[i,j] - bty[i,j]*vz[i,j]
            L[i,j,2,2] =  0
            L[i,j,2,3] = -btz[i,j]
            L[i,j,2,4] =  bty[i,j]
            L[i,j,2,5] = -btz[i,j]*sgnBx[i,j]*sqrt_rho[i,j]
            L[i,j,2,6] =  bty[i,j]*sgnBx[i,j]*sqrt_rho[i,j]
            L[i,j,2,7] =  0
            
            L[i,j,3,1] =  as[i,j]*(gamS*cs2[i,j]+ls[i,j]*vx[i,j]) + 
                          af[i,j]*lf[i,j]*(bty[i,j]*vy[i,j]+btz[i,j]*vz[i,j])*sgnBx[i,j]
            L[i,j,3,2] = -as[i,j]*ls[i,j]
            L[i,j,3,3] = -af[i,j]*lf[i,j]*bty[i,j]*sgnBx[i,j]
            L[i,j,3,4] = -af[i,j]*lf[i,j]*btz[i,j]*sgnBx[i,j]
            L[i,j,3,5] = -cs[i,j]*af[i,j]*bty[i,j]*sqrt_rho[i,j]
            L[i,j,3,6] = -cs[i,j]*af[i,j]*btz[i,j]*sqrt_rho[i,j]
            L[i,j,3,7] =  as[i,j]*cs2[i,j]*rho[i,j]/(gam*S[i,j])
            
            L[i,j,4,1] =  1/gam
            L[i,j,4,2] =  0
            L[i,j,4,3] =  0
            L[i,j,4,4] =  0
            L[i,j,4,5] =  0
            L[i,j,4,6] =  0
            L[i,j,4,7] = -rho[i,j]/(gam*S[i,j])
            
            L[i,j,5,1] =  as[i,j]*(gamS*cs2[i,j]-ls[i,j]*vx[i,j]) - 
                          af[i,j]*lf[i,j]*(bty[i,j]*vy[i,j]+btz[i,j]*vz[i,j])*sgnBx[i,j]
            L[i,j,5,2] =  as[i,j]*ls[i,j]
            L[i,j,5,3] =  af[i,j]*lf[i,j]*bty[i,j]*sgnBx[i,j]
            L[i,j,5,4] =  af[i,j]*lf[i,j]*btz[i,j]*sgnBx[i,j]
            L[i,j,5,5] = -cs[i,j]*af[i,j]*bty[i,j]*sqrt_rho[i,j]
            L[i,j,5,6] = -cs[i,j]*af[i,j]*btz[i,j]*sqrt_rho[i,j]
            L[i,j,5,7] =  as[i,j]*cs2[i,j]*rho[i,j]/(gam*S[i,j])
            
            L[i,j,6,1] =  btz[i,j]*vy[i,j] - bty[i,j]*vz[i,j]
            L[i,j,6,2] =  0
            L[i,j,6,3] = -btz[i,j]
            L[i,j,6,4] =  bty[i,j]
            L[i,j,6,5] =  btz[i,j]*sgnBx[i,j]*sqrt_rho[i,j]
            L[i,j,6,6] = -bty[i,j]*sgnBx[i,j]*sqrt_rho[i,j]
            L[i,j,6,7] =  0
            
            L[i,j,7,1] =  af[i,j]*(gamS*cs2[i,j]-lf[i,j]*vx[i,j]) + 
                          as[i,j]*ls[i,j]*(bty[i,j]*vy[i,j]+btz[i,j]*vz[i,j])*sgnBx[i,j]
            L[i,j,7,2] =  af[i,j]*lf[i,j]
            L[i,j,7,3] = -as[i,j]*ls[i,j]*bty[i,j]*sgnBx[i,j]
            L[i,j,7,4] = -as[i,j]*ls[i,j]*btz[i,j]*sgnBx[i,j]
            L[i,j,7,5] =  cs[i,j]*as[i,j]*bty[i,j]*sqrt_rho[i,j]
            L[i,j,7,6] =  cs[i,j]*as[i,j]*btz[i,j]*sqrt_rho[i,j]
            L[i,j,7,7] =  af[i,j]*cs2[i,j]*rho[i,j]/(gam*S[i,j])

	    	R[i,j,1,1] = af[i,j]
            R[i,j,2,1] = af[i,j]*(vx[i,j]-lf[i,j])
            R[i,j,3,1] = af[i,j]*vy[i,j] + as[i,j]*ls[i,j]*bty[i,j]*sgnBx[i,j]
            R[i,j,4,1] = af[i,j]*vz[i,j] + as[i,j]*ls[i,j]*btz[i,j]*sgnBx[i,j]
            R[i,j,5,1] = cs[i,j]*as[i,j]*bty[i,j]/sqrt_rho[i,j]
            R[i,j,6,1] = cs[i,j]*as[i,j]*btz[i,j]/sqrt_rho[i,j]
            R[i,j,7,1] = af[i,j]*S[i,j]/rho[i,j]
            
            R[i,j,1,2] =  0
            R[i,j,2,2] =  0
            R[i,j,3,2] = -btz[i,j]
            R[i,j,4,2] =  bty[i,j]
            R[i,j,5,2] = -btz[i,j]*sgnBx[i,j]/sqrt_rho[i,j]
            R[i,j,6,2] =  bty[i,j]*sgnBx[i,j]/sqrt_rho[i,j]
            R[i,j,7,2] =  0
            
            R[i,j,1,3] =  as[i,j]
            R[i,j,2,3] =  as[i,j]*(vx[i,j]-ls[i,j])
            R[i,j,3,3] =  as[i,j]*vy[i,j] - af[i,j]*lf[i,j]*bty[i,j]*sgnBx[i,j]
            R[i,j,4,3] =  as[i,j]*vz[i,j] - af[i,j]*lf[i,j]*btz[i,j]*sgnBx[i,j]
            R[i,j,5,3] = -cs[i,j]*af[i,j]*bty[i,j]/sqrt_rho[i,j]
            R[i,j,6,3] = -cs[i,j]*af[i,j]*btz[i,j]/sqrt_rho[i,j]
            R[i,j,7,3] =  as[i,j]*S[i,j]/rho[i,j]
            
            R[i,j,1,4] =  1
            R[i,j,2,4] =  vx[i,j]
            R[i,j,3,4] =  vy[i,j]
            R[i,j,4,4] =  vz[i,j]
            R[i,j,5,4] =  0
            R[i,j,6,4] =  0
            R[i,j,7,4] =  gam0*S[i,j]/rho[i,j]
            
            R[i,j,1,5] =  as[i,j]
            R[i,j,2,5] =  as[i,j]*(vx[i,j]+ls[i,j])
            R[i,j,3,5] =  as[i,j]*vy[i,j] + af[i,j]*lf[i,j]*bty[i,j]*sgnBx[i,j]
            R[i,j,4,5] =  as[i,j]*vz[i,j] + af[i,j]*lf[i,j]*btz[i,j]*sgnBx[i,j]
            R[i,j,5,5] = -cs[i,j]*af[i,j]*bty[i,j]/sqrt_rho[i,j]
            R[i,j,6,5] = -cs[i,j]*af[i,j]*btz[i,j]/sqrt_rho[i,j]
            R[i,j,7,5] =  as[i,j]*S[i,j]/rho[i,j]
            
            R[i,j,1,6] =  0
            R[i,j,2,6] =  0
            R[i,j,3,6] = -btz[i,j]
            R[i,j,4,6] =  bty[i,j]
            R[i,j,5,6] =  btz[i,j]*sgnBx[i,j]/sqrt_rho[i,j]
            R[i,j,6,6] = -bty[i,j]*sgnBx[i,j]/sqrt_rho[i,j]
            R[i,j,7,6] =  0
            
            R[i,j,1,7] = af[i,j]
            R[i,j,2,7] = af[i,j]*(vx[i,j]+lf[i,j])
            R[i,j,3,7] = af[i,j]*vy[i,j] - as[i,j]*ls[i,j]*bty[i,j]*sgnBx[i,j]
            R[i,j,4,7] = af[i,j]*vz[i,j] - as[i,j]*ls[i,j]*btz[i,j]*sgnBx[i,j]
            R[i,j,5,7] = cs[i,j]*as[i,j]*bty[i,j]/sqrt_rho[i,j]
            R[i,j,6,7] = cs[i,j]*as[i,j]*btz[i,j]/sqrt_rho[i,j]
            R[i,j,7,7] = af[i,j]*S[i,j]/rho[i,j]

      	    end  # for i
    end # for j
            
    @inbounds for m = 1:7
        @inbounds for j = 1:Nqy-1
    	    @inbounds @simd for i = 1:Nqx-1
                L[i,j,1,m] *= 0.5/cs2[i,j]
                L[i,j,2,m] *= 0.5
                L[i,j,3,m] *= 0.5/cs2[i,j] # 4 missing ?
                L[i,j,5,m] *= 0.5/cs2[i,j]
                L[i,j,6,m] *= 0.5
            	L[i,j,7,m] *= 0.5/cs2[i,j]
            end
        end
    end
    
	# enforce continuity

    sgnBt = sign.(By)
    bad = find(By .== 0)
    sgnBt[bad] = sign.(Bz[bad])
    bad = find(sgnBt .== 0)
    sgnBt[bad] = 1

    @inbounds for m=1:7
        @inbounds for j = 1:Nqy-1
	        @inbounds @simd for i = 1:Nqx-1

                if cs[i,j] >= la[i,j] 
		            L[i,j,3,m] *= sgnBt[i,j] 
        		    L[i,j,5,m] *= sgnBt[i,j] 
		            R[i,j,m,3] *= sgnBt[i,j] 
        		    R[i,j,m,5] *= sgnBt[i,j] 
            	else
        		    L[i,j,1,m] *= sgnBt[i,j] 
		            L[i,j,7,m] *= sgnBt[i,j] 
		            R[i,j,m,1] *= sgnBt[i,j] 
		            R[i,j,m,7] *= sgnBt[i,j] 
	            end
            end 
        end
    end

    return L, R
end # compute_eigenvectors_S


function compute_fluxes_S(q::Array{Float64,3}, u::Array{Float64,3})
	
    const Nux = size(u,1)
    const Nuy = size(u,2)

	F = zeros(Float64, Nux, Nuy, 7)

    @inbounds for j = 1:Nuy
        @inbounds @simd for i = 1:Nux

            pt = u[i,j,8] + 0.5*(u[i,j,5]^2+u[i,j,6]^2+u[i,j,7]^2) 
    
            F[i,j,1] = u[i,j,1]*u[i,j,2] 				        # rho * vx
            F[i,j,2] = q[i,j,2]*u[i,j,2] + pt - u[i,j,5]^2 	    # mvx*vx + Ptot - Bx^2
            F[i,j,3] = q[i,j,3]*u[i,j,2] - u[i,j,5]*u[i,j,6]    # mvy*vx - Bx*By
            F[i,j,4] = q[i,j,4]*u[i,j,2] - u[i,j,5]*u[i,j,7] 	# mvy*vx - Bx*Bz
            F[i,j,5] = u[i,j,6]*u[i,j,2] - u[i,j,5]*u[i,j,3] 	# By*vx - Bx*vy
            F[i,j,6] = u[i,j,7]*u[i,j,2] - u[i,j,5]*u[i,j,4]    # Bz*vx - Bx*vz
            F[i,j,7] = q[i,j,8]*u[i,j,2]	               	    # S*vx
        end
    end

	return F # cell centered fluxes
end


function compute_primitive_variables_S(q::Array{Float64,3})

	u = copy(q)  # = [rho,Mx,My,Mz,Bx,By,Bz,S]

    const gM1 = gam-1

    @inbounds for j = 1:size(u,2)
        @inbounds @simd for i = 1:size(u,1)
            u[i,j,2] /= u[i,j,1]   # vx = Mx/rho
            u[i,j,3] /= u[i,j,1]   # vy
            u[i,j,4] /= u[i,j,1]   # vz
            u[i,j,8] *= u[i,j,1]^gM1    # Pressure from S, doesnt vectorize
        end
    end

	return u # = [rho,vx,vy,vz,Bx,By,Bz,P(S)]; J&W eq. 2.23
end

# Energy (E) - code

function weno5_flux_difference_E(q::Array{Float64,3})

    const Nqx = size(q,1) # q might be rotated
    const Nqy = size(q,2)

    u = compute_primitive_variables_E(q) # [rho,vx,vy,vz,Bx,By,Bz,P(S)]  

    a = compute_eigenvalues(u)

    F = compute_fluxes_E(q,u)

    #L, R = compute_eigenvectors_E(q,u,F)
    L, R = compute_eigenvectors_E_vec(q,u,F,gam)
   
    dF = weno5_interpolation(q,a,F,L,R)

    dq =  zeros(Float64,Nqx, Nqy, 8)
    bsy = zeros(Float64,Nqx, Nqy)
    bsz = zeros(Float64,Nqx, Nqy)

    @inbounds for j = NBnd:Nqy-NBnd+1
        @inbounds @simd for i = NBnd:Nqx-NBnd+1
    		dq[i,j,1] = dF[i,j,1] - dF[i-1,j,1]
    		dq[i,j,2] = dF[i,j,2] - dF[i-1,j,2]
    		dq[i,j,3] = dF[i,j,3] - dF[i-1,j,3]
    		dq[i,j,4] = dF[i,j,4] - dF[i-1,j,4]
    		dq[i,j,5] = 0                   # fluxct scheme later
    		dq[i,j,6] = dF[i,j,5] - dF[i-1,j,5]
    		dq[i,j,7] = dF[i,j,6] - dF[i-1,j,6]
	    	dq[i,j,8] = dF[i,j,7] - dF[i-1,j,7]
	    end

        @inbounds @simd for i = NBnd:Nqx-NBnd+1
            bsy[i,j] = dF[i,j,5] + 0.5 *(u[i,j,5]*u[i,j,3] + u[i+1,j,5]*u[i+1,j,3])
            bsz[i,j] = dF[i,j,6] + 0.5 *(u[i,j,5]*u[i,j,4] + u[i+1,j,5]*u[i+1,j,4])
        end
    end
        
    return dq,  bsy, bsz
end

function compute_eigenvectors_E(q::Array{Float64,3}, u::Array{Float64,3}, 
								F::Array{Float64,3}) # Roe solver
    const gam0 = 1-gam
    const gam1 = (gam-1)/2
    const gam2 = (gam-2)/(gam-1)

    const Nqx = size(q,1)
    const Nqy = size(q,2)

	L = zeros(Float64, Nqx, Nqy, 7,7) # left hand eigenvectors
	R = zeros(Float64, Nqx, Nqy, 7,7) # right hand eigenvectors

	@inbounds for j = 1:Nqy-1
	    @inbounds for i = 1:Nqx-1
		
		    rho = 0.5 * (u[i,j,1] + u[i+1,j,1]) # to cell boundary

		    vx = 0.5 * (u[i,j,2] + u[i+1,j,2])
		    vy = 0.5 * (u[i,j,3] + u[i+1,j,3])
		    vz = 0.5 * (u[i,j,4] + u[i+1,j,4])

            Bx = 0.5 * (u[i,j,5] + u[i+1,j,5])
		    By = 0.5 * (u[i,j,6] + u[i+1,j,6])
		    Bz = 0.5 * (u[i,j,7] + u[i+1,j,7])
		
            E = 0.5 * (q[i,j,8] + q[i+1,j,8]) # q NOT u

		    v2 = vx^2 + vy^2 + vz^2
		    B2 = Bx^2 + By^2 + Bz^2

		    pg = (gam-1) * (E - 0.5 * (rho*v2 + B2) )

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
    
            if sgnBx == 0
                sgnBx = 1
            end

		    if Bt2 >= 1e-30
		    	bty = By/sqrt(Bt2)
		    	btz = Bz/sqrt(Bt2)
            else
		        bty = 1/sqrt(2)
		        btz = 1/sqrt(2)
		    end

		    dl2 = lf^2 - ls^2

		    if dl2 >= 1e-30
                af = sqrt(max(0,cs2 - ls^2)/dl2)
		    	as = sqrt(max(0,lf^2 - cs2)/dl2)
            else
		        af = 1.0
		        as = 1.0
		    end

		    # eigenvectors E version

		    sqrt_rho = sqrt(rho)

		    L[i,j,1,1] = af*(gam1*v2 + lf*vx) - as*ls*(bty*vy + btz*vz)*sgnBx
            L[i,j,1,2] = af*(gam0*vx - lf)
            L[i,j,1,3] = gam0*af*vy + as*ls*bty*sgnBx
            L[i,j,1,4] = gam0*af*vz + as*ls*btz*sgnBx
            L[i,j,1,5] = gam0*af*By + cs*as*bty*sqrt_rho
            L[i,j,1,6] = gam0*af*Bz + cs*as*btz*sqrt_rho
            L[i,j,1,7] = -gam0*af
                                                                   
            L[i,j,2,1] = btz*vy - bty*vz
            L[i,j,2,2] = 0
            L[i,j,2,3] = -btz
            L[i,j,2,4] = bty
            L[i,j,2,5] = -btz*sgnBx*sqrt_rho
            L[i,j,2,6] = bty*sgnBx*sqrt_rho
            L[i,j,2,7] = 0

            L[i,j,3,1] = as*(gam1*v2 + ls*vx) + af*lf*(bty*vy + btz*vz)*sgnBx
            L[i,j,3,2] = gam0*as*vx - as*ls
            L[i,j,3,3] = gam0*as*vy - af*lf*bty*sgnBx
            L[i,j,3,4] = gam0*as*vz - af*lf*btz*sgnBx
            L[i,j,3,5] = gam0*as*By - cs*af*bty*sqrt_rho 
            L[i,j,3,6] = gam0*as*Bz - cs*af*btz*sqrt_rho 
            L[i,j,3,7] = -gam0*as
                                                                   
            L[i,j,4,1] = -cs2/gam0 - 0.5*v2
            L[i,j,4,2] = vx
            L[i,j,4,3] = vy
            L[i,j,4,4] = vz
            L[i,j,4,5] = By
            L[i,j,4,6] = Bz
            L[i,j,4,7] = -1
                                                                   
            L[i,j,5,1] = as*(gam1*v2 - ls*vx) - af*lf*(bty*vy + btz*vz)*sgnBx
            L[i,j,5,2] = as*(gam0*vx+ls)
            L[i,j,5,3] = gam0*as*vy + af*lf*bty*sgnBx
            L[i,j,5,4] = gam0*as*vz + af*lf*btz*sgnBx
            L[i,j,5,5] = gam0*as*By - cs*af*bty*sqrt_rho 
            L[i,j,5,6] = gam0*as*Bz - cs*af*btz*sqrt_rho 
            L[i,j,5,7] = -gam0*as
                                                                   
            L[i,j,6,1] = btz*vy - bty*vz
            L[i,j,6,2] = 0
            L[i,j,6,3] = -btz
            L[i,j,6,4] = bty
            L[i,j,6,5] = btz*sgnBx*sqrt_rho
            L[i,j,6,6] = -bty*sgnBx*sqrt_rho
            L[i,j,6,7] = 0
                                                                   
            L[i,j,7,1] = af*(gam1*v2 - lf*vx) + as*ls*(bty*vy + btz*vz)*sgnBx
            L[i,j,7,2] = af*(gam0*vx + lf)
            L[i,j,7,3] = gam0*af*vy - as*ls*bty*sgnBx
            L[i,j,7,4] = gam0*af*vz - as*ls*btz*sgnBx
            L[i,j,7,5] = gam0*af*By + cs*as*bty*sqrt_rho
            L[i,j,7,6] = gam0*af*Bz + cs*as*btz*sqrt_rho
            L[i,j,7,7] = -gam0*af

            for m=1:7
                L[i,j,1,m] *= 0.5/cs2
                L[i,j,2,m] *= 0.5
                L[i,j,3,m] *= 0.5/cs2 
			    L[i,j,4,m] *= -gam0/cs2
                L[i,j,5,m] *= 0.5/cs2
                L[i,j,6,m] *= 0.5
        	    L[i,j,7,m] *= 0.5/cs2
            end

		    R[i,j,1,1] = af
            R[i,j,2,1] = af*(vx - lf)
            R[i,j,3,1] = af*vy + as*ls*bty*sgnBx
            R[i,j,4,1] = af*vz + as*ls*btz*sgnBx
            R[i,j,5,1] = cs*as*bty/sqrt_rho
            R[i,j,6,1] = cs*as*btz/sqrt_rho
            R[i,j,7,1] = af*(lf^2 - lf*vx + 0.5*v2 - gam2*cs2) + as*ls*(bty*vy + btz*vz)*sgnBx 
           
            R[i,j,1,2] = 0
            R[i,j,2,2] = 0
            R[i,j,3,2] = -btz
            R[i,j,4,2] = bty
            R[i,j,5,2] = -btz*sgnBx/sqrt_rho
            R[i,j,6,2] = bty*sgnBx/sqrt_rho
            R[i,j,7,2] = bty*vz - btz*vy
                                                      
            R[i,j,1,3] = as
            R[i,j,2,3] = as*(vx-ls)
            R[i,j,3,3] = as*vy - af*lf*bty*sgnBx
            R[i,j,4,3] = as*vz - af*lf*btz*sgnBx
            R[i,j,5,3] = -cs*af*bty/sqrt_rho
            R[i,j,6,3] = -cs*af*btz/sqrt_rho
            R[i,j,7,3] = as*(ls^2 - ls*vx + 0.5*v2 - gam2*cs2) - af*lf*(bty*vy + btz*vz)*sgnBx
            
            R[i,j,1,4] = 1
            R[i,j,2,4] = vx
            R[i,j,3,4] = vy
            R[i,j,4,4] = vz
            R[i,j,5,4] = 0
            R[i,j,6,4] = 0
            R[i,j,7,4] = 0.5*v2
                                                       
            R[i,j,1,5] =  as
            R[i,j,2,5] =  as*(vx + ls)
            R[i,j,3,5] =  as*vy + af*lf*bty*sgnBx
            R[i,j,4,5] =  as*vz + af*lf*btz*sgnBx
            R[i,j,5,5] = -cs*af*bty/sqrt_rho
            R[i,j,6,5] = -cs*af*btz/sqrt_rho
            R[i,j,7,5] =  as*(ls^2 + ls*vx + 0.5*v2 - gam2*cs2) + af*lf*(bty*vy + btz*vz)*sgnBx
            
            R[i,j,1,6] =  0
            R[i,j,2,6] =  0
            R[i,j,3,6] = -btz
            R[i,j,4,6] =  bty
            R[i,j,5,6] =  btz*sgnBx/sqrt_rho
            R[i,j,6,6] = -bty*sgnBx/sqrt_rho
            R[i,j,7,6] =  bty*vz - btz*vy
                                                      
            R[i,j,1,7] = af
            R[i,j,2,7] = af*(vx + lf)
            R[i,j,3,7] = af*vy - as*ls*bty*sgnBx
            R[i,j,4,7] = af*vz - as*ls*btz*sgnBx
            R[i,j,5,7] = cs*as*bty/sqrt_rho
            R[i,j,6,7] = cs*as*btz/sqrt_rho
            R[i,j,7,7] = af*(lf^2 + lf*vx + 0.5*v2 - gam2*cs2) - as*ls*(bty*vy + btz*vz)*sgnBx

		    # enforce continuity

		    if By != 0
			    sgnBt = sign(By)
            else
                sgnBt = sign(Bz)
		    end

            if sgnBt == 0
                sgnBt = 1
            end

		    if cs >= la
			    for m=1:7
				    L[i,j,3,m] *= sgnBt 
				    L[i,j,5,m] *= sgnBt 
				    R[i,j,m,3] *= sgnBt 
				    R[i,j,m,5] *= sgnBt 
			    end
		    else
			    for m=1:7
				    L[i,j,1,m] *= sgnBt 
				    L[i,j,7,m] *= sgnBt 
				    R[i,j,m,1] *= sgnBt 
				    R[i,j,m,7] *= sgnBt 
			    end
		    end
	    end  # for i
	end  # for j

	return L, R
end # compute_eigenvectors_E



function compute_eigenvectors_E_vec(q::Array{Float64,3}, u::Array{Float64,3}, 
								F::Array{Float64,3}, gam::Float64) # Roe solver
    const gM1 = gam-1
    const gam0 = 1-gam
    const gam1 = (gam-1)/2
    const gam2 = (gam-2)/(gam-1)

    const Nqx = size(q,1)
    const Nqy = size(q,2)

    rho = zeros(Float64, Nqx, Nqy)
    sqrt_rho = zeros(Float64, Nqx, Nqy)
    vx = zeros(Float64, Nqx, Nqy)
    vy = zeros(Float64, Nqx, Nqy)
    vz = zeros(Float64, Nqx, Nqy)
    Bx = zeros(Float64, Nqx, Nqy)
    By = zeros(Float64, Nqx, Nqy)
    Bz = zeros(Float64, Nqx, Nqy)
    E = zeros(Float64, Nqx, Nqy)

	@inbounds for j = 1:Nqy-1
	    @inbounds @simd for i = 1:Nqx-1
		
            rho[i,j] = 0.5 * (u[i,j,1] + u[i+1,j,1]) # to cell boundary
            
            sqrt_rho[i,j] = sqrt(rho[i,j])

		    vx[i,j] = 0.5 * (u[i,j,2] + u[i+1,j,2])
		    vy[i,j] = 0.5 * (u[i,j,3] + u[i+1,j,3])
		    vz[i,j] = 0.5 * (u[i,j,4] + u[i+1,j,4])

            Bx[i,j] = 0.5 * (u[i,j,5] + u[i+1,j,5])
		    By[i,j] = 0.5 * (u[i,j,6] + u[i+1,j,6])
		    Bz[i,j] = 0.5 * (u[i,j,7] + u[i+1,j,7])
		
            E[i,j] = 0.5 * (q[i,j,8] + q[i+1,j,8]) # q NOT u
        end
    end

    v2 = Array{Float64}( Nqx, Nqy)
    pg = Array{Float64}( Nqx, Nqy)
    bbn2 = Array{Float64}( Nqx, Nqy)

	@inbounds for j = 1:Nqy
	    @inbounds @simd for i = 1:Nqx

            B2 = (Bx[i,j]^2 + By[i,j]^2 + Bz[i,j]^2)

            v2[i,j] = vx[i,j]^2 + vy[i,j]^2 + vz[i,j]^2
	        pg[i,j] = (gM1) * (E[i,j] - 0.5 * (rho[i,j]*v2[i,j] + B2) )
            bbn2[i,j] = B2[i,j]/rho[i,j]
        end
    end

    cs2 = Array{Float64}(Nqx, Nqy)
    bnx2 = Array{Float64}(Nqx, Nqy)

	@inbounds for j = 1:Nqy
	    @inbounds @simd for i = 1:Nqx

	        cs2[i,j] = max(0, gam*abs(pg[i,j]/rho[i,j]))
	        bnx2[i,j] = Bx[i,j]^2 / rho[i,j]
        end
    end

    cs =  Array{Float64}(Nqx, Nqy)
    root =  Array{Float64}(Nqx, Nqy)
    lf =  Array{Float64}(Nqx, Nqy) # fast mode
    la =  Array{Float64}(Nqx, Nqy) # alven mode
    ls =  Array{Float64}(Nqx, Nqy) # slow mode
    Bt2 =  Array{Float64}(Nqx, Nqy)

    @. cs = sqrt(cs2)
    @. root = sqrt.(max.(0, (bbn2+cs2)^2 - 4*bnx2*cs2))
    @. lf = sqrt.(max.(0, (bbn2+cs2+root)/2)) 	
    @. la = sqrt.(max.(0, bnx2))					        
	@. ls = sqrt.(max.(0, (bbn2+cs2-root)/2))	
    @. Bt2 = By^2 + Bz^2

	# resolve degeneracies
	
    sgnBx =  Array{Float64}(Nqx, Nqy)
    @. sgnBx = sign(Bx) # -1, 0, 1
    sgnBx[find(sgnBx .== 0)] = 1

    bty =  Array{Float64}(Nqx, Nqy)
    btz =  Array{Float64}(Nqx, Nqy)
    @. bty = By/sqrt(Bt2)
    @. btz = Bz/sqrt(Bt2)
    bad = find(Bt2 .<= 1e-30)
    bty[bad] = 1/sqrt(2)
    btz[bad] = 1/sqrt(2)

    dl2 =  Array{Float64}(Nqx, Nqy)
    @. dl2 = lf^2 - ls^2

    af = fill(1.0, Nqx, Nqy)
    as = fill(1.0, Nqx, Nqy)

    @inbounds for j = 1:Nqy
	    @inbounds @simd for i = 1:Nqx-1

            if dl2[i,j] >= 1e-30
                af[i,j] = sqrt(max(0, cs2[i,j] - ls[i,j].^2)./dl2[i,j])
                as[i,j] = sqrt(max(0, lf[i,j].^2 - cs2[i,j])./dl2[i,j])
            end
        end
    end

    L = Array{Float64}( Nqx, Nqy, 7,7) # left hand eigenvectors
    R = Array{Float64}( Nqx, Nqy, 7,7) # right hand eigenvectors

    @inbounds for j = 1:Nqy-1 # eigenvectors E version
	    @inbounds @simd for i = 1:Nqx-1

            L[i,j,1,1] = af[i,j] * (gam1*v2[i,j] + lf[i,j]*vx[i,j]) - 
                         as[i,j]*ls[i,j]*(bty[i,j]*vy[i,j] + btz[i,j]*vz[i,j]) * sgnBx[i,j]
            L[i,j,1,2] = af[i,j]*(gam0*vx[i,j] - lf[i,j])
            L[i,j,1,3] = gam0*af[i,j]*vy[i,j] + as[i,j]*ls[i,j]*bty[i,j]*sgnBx[i,j]
            L[i,j,1,4] = gam0*af[i,j]*vz[i,j] + as[i,j]*ls[i,j]*btz[i,j]*sgnBx[i,j]
            L[i,j,1,5] = gam0*af[i,j]*By[i,j] + cs[i,j]*as[i,j]*bty[i,j]*sqrt_rho[i,j]
            L[i,j,1,6] = gam0*af[i,j]*Bz[i,j] + cs[i,j]*as[i,j]*btz[i,j]*sqrt_rho[i,j]
            L[i,j,1,7] = -gam0*af[i,j]
      
            L[i,j,2,1] = btz[i,j]*vy[i,j] - bty[i,j]*vz[i,j]
            L[i,j,2,2] = 0
            L[i,j,2,3] = -btz[i,j]
            L[i,j,2,4] = bty[i,j]
            L[i,j,2,5] = -btz[i,j]*sgnBx[i,j]*sqrt_rho[i,j]
            L[i,j,2,6] = bty[i,j]*sgnBx[i,j]*sqrt_rho[i,j]
            L[i,j,2,7] = 0

            L[i,j,3,1] = as[i,j]*(gam1*v2[i,j] + ls[i,j]*vx[i,j]) + 
                         af[i,j]*lf[i,j]*(bty[i,j]*vy[i,j] + btz[i,j]*vz[i,j])*sgnBx[i,j]
            L[i,j,3,2] = gam0*as[i,j]*vx[i,j] - as[i,j]*ls[i,j]
            L[i,j,3,3] = gam0*as[i,j]*vy[i,j] - af[i,j]*lf[i,j]*bty[i,j]*sgnBx[i,j]
            L[i,j,3,4] = gam0*as[i,j]*vz[i,j] - af[i,j]*lf[i,j]*btz[i,j]*sgnBx[i,j]
            L[i,j,3,5] = gam0*as[i,j]*By[i,j] - cs[i,j]*af[i,j]*bty[i,j]*sqrt_rho[i,j]
            L[i,j,3,6] = gam0*as[i,j]*Bz[i,j] - cs[i,j]*af[i,j]*btz[i,j]*sqrt_rho[i,j]
            L[i,j,3,7] = -gam0*as[i,j]
                                                                   
            L[i,j,4,1] = -cs2[i,j]/gam0 - 0.5*v2[i,j]
            L[i,j,4,2] = vx[i,j]
            L[i,j,4,3] = vy[i,j]
            L[i,j,4,4] = vz[i,j]
            L[i,j,4,5] = By[i,j]
            L[i,j,4,6] = Bz[i,j]
            L[i,j,4,7] = -1
                                                                   
            L[i,j,5,1] = as[i,j]*(gam1*v2[i,j] - ls[i,j]*vx[i,j]) - 
                         af[i,j]*lf[i,j]*(bty[i,j]*vy[i,j] + btz[i,j]*vz[i,j])*sgnBx[i,j]
            L[i,j,5,2] = as[i,j]*(gam0*vx[i,j]+ls[i,j])
            L[i,j,5,3] = gam0*as[i,j]*vy[i,j] + af[i,j]*lf[i,j]*bty[i,j]*sgnBx[i,j]
            L[i,j,5,4] = gam0*as[i,j]*vz[i,j] + af[i,j]*lf[i,j]*btz[i,j]*sgnBx[i,j]
            L[i,j,5,5] = gam0*as[i,j]*By[i,j] - cs[i,j]*af[i,j]*bty[i,j]*sqrt_rho[i,j]
            L[i,j,5,6] = gam0*as[i,j]*Bz[i,j] - cs[i,j]*af[i,j]*btz[i,j]*sqrt_rho[i,j]
            L[i,j,5,7] = -gam0*as[i,j]
                                                                   
            L[i,j,6,1] = btz[i,j]*vy[i,j] - bty[i,j]*vz[i,j]
            L[i,j,6,2] = 0
            L[i,j,6,3] = -btz[i,j]
            L[i,j,6,4] = bty[i,j]
            L[i,j,6,5] = btz[i,j]*sgnBx[i,j]*sqrt_rho[i,j]
            L[i,j,6,6] = -bty[i,j]*sgnBx[i,j]*sqrt_rho[i,j]
            L[i,j,6,7] = 0
                                                                   
            L[i,j,7,1] = af[i,j]*(gam1*v2[i,j] - lf[i,j]*vx[i,j]) + 
                         as[i,j]*ls[i,j]*(bty[i,j]*vy[i,j] + btz[i,j]*vz[i,j])*sgnBx[i,j]
            L[i,j,7,2] = af[i,j]*(gam0*vx[i,j] + lf[i,j])
            L[i,j,7,3] = gam0*af[i,j]*vy[i,j] - as[i,j]*ls[i,j]*bty[i,j]*sgnBx[i,j]
            L[i,j,7,4] = gam0*af[i,j]*vz[i,j] - as[i,j]*ls[i,j]*btz[i,j]*sgnBx[i,j]
            L[i,j,7,5] = gam0*af[i,j]*By[i,j] + cs[i,j]*as[i,j]*bty[i,j]*sqrt_rho[i,j]
            L[i,j,7,6] = gam0*af[i,j]*Bz[i,j] + cs[i,j]*as[i,j]*btz[i,j]*sqrt_rho[i,j]
            L[i,j,7,7] = -gam0*af[i,j]

		    R[i,j,1,1] = af[i,j]
            R[i,j,2,1] = af[i,j]*(vx[i,j] - lf[i,j])
            R[i,j,3,1] = af[i,j]*vy[i,j] + as[i,j]*ls[i,j]*bty[i,j]*sgnBx[i,j]
            R[i,j,4,1] = af[i,j]*vz[i,j] + as[i,j]*ls[i,j]*btz[i,j]*sgnBx[i,j]
            R[i,j,5,1] = cs[i,j]*as[i,j]*bty[i,j]/sqrt_rho[i,j]
            R[i,j,6,1] = cs[i,j]*as[i,j]*btz[i,j]/sqrt_rho[i,j]
            R[i,j,7,1] = af[i,j]*(lf[i,j]^2 - lf[i,j]*vx[i,j] + 0.5*v2[i,j] - gam2*cs2[i,j]) +
                         as[i,j]*ls[i,j]*(bty[i,j]*vy[i,j] + btz[i,j]*vz[i,j])*sgnBx[i,j] 
           
            R[i,j,1,2] = 0
            R[i,j,2,2] = 0
            R[i,j,3,2] = -btz[i,j]
            R[i,j,4,2] = bty[i,j]
            R[i,j,5,2] = -btz[i,j]*sgnBx[i,j]/sqrt_rho[i,j]
            R[i,j,6,2] = bty[i,j]*sgnBx[i,j]/sqrt_rho[i,j]
            R[i,j,7,2] = bty[i,j]*vz[i,j] - btz[i,j]*vy[i,j]
                                                      
            R[i,j,1,3] = as[i,j]
            R[i,j,2,3] = as[i,j]*(vx[i,j]-ls[i,j])
            R[i,j,3,3] = as[i,j]*vy[i,j] - af[i,j]*lf[i,j]*bty[i,j]*sgnBx[i,j]
            R[i,j,4,3] = as[i,j]*vz[i,j] - af[i,j]*lf[i,j]*btz[i,j]*sgnBx[i,j]
            R[i,j,5,3] = -cs[i,j]*af[i,j]*bty[i,j]/sqrt_rho[i,j]
            R[i,j,6,3] = -cs[i,j]*af[i,j]*btz[i,j]/sqrt_rho[i,j]
            R[i,j,7,3] = as[i,j]*(ls[i,j]^2 - ls[i,j]*vx[i,j] + 0.5*v2[i,j] - gam2*cs2[i,j]) - 
                         af[i,j]*lf[i,j]*(bty[i,j]*vy[i,j] + btz[i,j]*vz[i,j])*sgnBx[i,j]
            
            R[i,j,1,4] = 1
            R[i,j,2,4] = vx[i,j]
            R[i,j,3,4] = vy[i,j]
            R[i,j,4,4] = vz[i,j]
            R[i,j,5,4] = 0
            R[i,j,6,4] = 0
            R[i,j,7,4] = 0.5*v2[i,j]
                                                       
            R[i,j,1,5] =  as[i,j]
            R[i,j,2,5] =  as[i,j]*(vx[i,j] + ls[i,j])
            R[i,j,3,5] =  as[i,j]*vy[i,j] + af[i,j]*lf[i,j]*bty[i,j]*sgnBx[i,j]
            R[i,j,4,5] =  as[i,j]*vz[i,j] + af[i,j]*lf[i,j]*btz[i,j]*sgnBx[i,j]
            R[i,j,5,5] = -cs[i,j]*af[i,j]*bty[i,j]/sqrt_rho[i,j]
            R[i,j,6,5] = -cs[i,j]*af[i,j]*btz[i,j]/sqrt_rho[i,j]
            R[i,j,7,5] =  as[i,j]*(ls[i,j]^2 + ls[i,j]*vx[i,j] + 0.5*v2[i,j] - gam2*cs2[i,j]) + 
                          af[i,j]*lf[i,j]*(bty[i,j]*vy[i,j] + btz[i,j]*vz[i,j])*sgnBx[i,j]
            
            R[i,j,1,6] =  0
            R[i,j,2,6] =  0
            R[i,j,3,6] = -btz[i,j]
            R[i,j,4,6] =  bty[i,j]
            R[i,j,5,6] =  btz[i,j]*sgnBx[i,j]/sqrt_rho[i,j]
            R[i,j,6,6] = -bty[i,j]*sgnBx[i,j]/sqrt_rho[i,j]
            R[i,j,7,6] =  bty[i,j]*vz[i,j] - btz[i,j]*vy[i,j]
                                                      
            R[i,j,1,7] = af[i,j]
            R[i,j,2,7] = af[i,j]*(vx[i,j] + lf[i,j])
            R[i,j,3,7] = af[i,j]*vy[i,j] - as[i,j]*ls[i,j]*bty[i,j]*sgnBx[i,j]
            R[i,j,4,7] = af[i,j]*vz[i,j] - as[i,j]*ls[i,j]*btz[i,j]*sgnBx[i,j]
            R[i,j,5,7] = cs[i,j]*as[i,j]*bty[i,j]/sqrt_rho[i,j]
            R[i,j,6,7] = cs[i,j]*as[i,j]*btz[i,j]/sqrt_rho[i,j]
            R[i,j,7,7] = af[i,j]*(lf[i,j]^2 + lf[i,j]*vx[i,j] + 0.5*v2[i,j] - gam2*cs2[i,j]) - 
                         as[i,j]*ls[i,j]*(bty[i,j]*vy[i,j] + btz[i,j]*vz[i,j])*sgnBx[i,j]
	    end  # for i
	end  # for j

    @inbounds for m = 1:7
        @inbounds for j = 1:Nqy-1
    	    @inbounds @simd for i = 1:Nqx-1
                L[i,j,1,m] *= 0.5/cs2[i,j]
                L[i,j,2,m] *= 0.5
                L[i,j,3,m] *= 0.5/cs2[i,j]
		    	L[i,j,4,m] *= -gam0/cs2[i,j]
                L[i,j,5,m] *= 0.5/cs2[i,j]
                L[i,j,6,m] *= 0.5
            	L[i,j,7,m] *= 0.5/cs2[i,j]
            end
        end
    end

	# enforce continuity

	sgnBt = sign.(By)
    bad = find(By .== 0)
    sgnBt[bad] = sign.(Bz[bad])
    bad = find(sgnBt .== 0)
    sgnBt[bad] = 1

    @inbounds for m=1:7
        @inbounds for j = 1:Nqy-1
	        @inbounds @simd for i = 1:Nqx-1
                if cs[i,j] >= la[i,j] 
		            L[i,j,3,m] *= sgnBt[i,j] 
        		    L[i,j,5,m] *= sgnBt[i,j] 
		            R[i,j,m,3] *= sgnBt[i,j] 
        		    R[i,j,m,5] *= sgnBt[i,j] 
            	else
        		    L[i,j,1,m] *= sgnBt[i,j] 
		            L[i,j,7,m] *= sgnBt[i,j] 
		            R[i,j,m,1] *= sgnBt[i,j] 
		            R[i,j,m,7] *= sgnBt[i,j] 
	            end
            end 
        end
    end

	return L, R
end # compute_eigenvectors_E

function compute_fluxes_E(q::Array{Float64,3}, u::Array{Float64,3})
	
    const Nux = size(u,1)
    const Nuy = size(u,2)

    F = Array{Float64}(Nux, Nuy, 7)

    @inbounds for j = 1:Nuy
        @inbounds @simd for i = 1:Nux

	        pt = u[i,j,8] + 0.5*(u[i,j,5]^2+u[i,j,6]^2+u[i,j,7]^2) # total pressure from E
    	    vB = u[i,j,2]*u[i,j,5] + u[i,j,3]*u[i,j,6] + u[i,j,4]*u[i,j,7] # v*B

            F[i,j,1] = u[i,j,1]*u[i,j,2] 					    # rho * vx
	        F[i,j,2] = q[i,j,2]*u[i,j,2] + pt - u[i,j,5]^2	    # mvx*vx + Ptot - Bx^2
            F[i,j,3] = q[i,j,3]*u[i,j,2] - u[i,j,5]*u[i,j,6] 	# mvy*vx - Bx*By
            F[i,j,4] = q[i,j,4]*u[i,j,2] - u[i,j,5]*u[i,j,7] 	# mvy*vx - Bx*Bz
            F[i,j,5] = u[i,j,6]*u[i,j,2] - u[i,j,5]*u[i,j,3] 	# By*vx - Bx*vy
            F[i,j,6] = u[i,j,7]*u[i,j,2] - u[i,j,5]*u[i,j,4]   	# Bz*vx - Bx*vz
            F[i,j,7] = (q[i,j,8]+pt)*u[i,j,2] - u[i,j,5]*vB	    # (E+Ptot)*vx - Bx * (v*B)
        end
    end

	return F
end
function compute_primitive_variables_E(q::Array{Float64,3})

    const gM1 = gam-1

    const Nqx = size(q,1)
    const Nqy = size(q,2)

    u = Array{Float64}(Nqx,Nqy,8) # = [rho,Mx,My,Mz,Bx,By,Bz,E]
	
    @inbounds for j = 1:Nqy
        @inbounds @simd for i = 1:Nqx
        
            u[i,j,1] = q[i,j,1]          # rho
            u[i,j,2] = q[i,j,2]/q[i,j,1] # vx = Mx/rho
    	    u[i,j,3] = q[i,j,3]/q[i,j,1] # vy
        	u[i,j,4] = q[i,j,4]/q[i,j,1] # vz
            u[i,j,5] = q[i,j,5]          # Bx
            u[i,j,6] = q[i,j,6]          # By
            u[i,j,7] = q[i,j,7]          # Bz
        end
    end

    @inbounds for j = 1:Nqy
        @inbounds @simd for i = 1:Nqx

    	    v2 = u[i,j,2]^2 + u[i,j,3]^2 + u[i,j,4]^2
	        B2 = u[i,j,5]^2 + u[i,j,6]^2 + u[i,j,7]^2

            u[i,j,8] = gM1 * (q[i,j,8] - 0.5 * (q[i,j,1]*v2 + B2)) # Pressure from E
        end
    end

	return u # = [rho,vx,vy,vz,Bx,By,Bz,P(E)] ; B&W eq. 2.23
end

function compute_eigenvalues(u::Array{Float64,3})

    const Nux = size(u, 1)
    const Nuy = size(u, 2)

    B2 = Array{Float64}(Nux, Nuy)
    vdB = Array{Float64}(Nux, Nuy)
    cs2 = Array{Float64}(Nux, Nuy)
    bnx2 = Array{Float64}(Nux, Nuy)

    @inbounds for j = 1:Nuy 
        @inbounds @simd for i = 1:Nux

            B2[i,j]  = u[i,j,5]^2 + u[i,j,6]^2 + u[i,j,7]^2
            vdB[i,j] = u[i,j,2]*u[i,j,5] + u[i,j,3]*u[i,j,6] + u[i,j,4]*u[i,j,7]
            cs2[i,j] = max(0,gam * abs(u[i,j,8]/u[i,j,1]))
            bnx2[i,j] = max(0, u[i,j,5]^2/u[i,j,1])
        end
    end

    bbn2 = Array{Float64}(Nux, Nuy)
    @. bbn2 = B2/u[:,:,1]

    root = Array{Float64}(Nux, Nuy)

    @inbounds for j = 1:Nuy 
        @inbounds @simd for i = 1:Nux    
            root[i,j] = sqrt(max(0,(bbn2[i,j]+cs2[i,j])^2 - 4*bnx2[i,j]*cs2[i,j]))
        end
    end

   lf = Array{Float64}(Nux, Nuy)
   ls = Array{Float64}(Nux, Nuy)
   la = Array{Float64}(Nux, Nuy)

   @inbounds for j = 1:Nuy 
       @inbounds @simd for i = 1:Nux
           lf[i,j] = sqrt(max(0, 0.5*(bbn2[i,j] + cs2[i,j] + root[i,j])))
           ls[i,j] = sqrt(max(0, 0.5*(bbn2[i,j] + cs2[i,j] - root[i,j])))
       end
   end

   @. la = sqrt(bnx2)

   a = Array{Float64}(Nux, Nuy, 7) # eigenvalues cell center

   @inbounds for j = 1:Nuy 
        @inbounds @simd for i = 1:Nux 
            a[i,j,1] = u[i,j,2] - lf[i,j] # u is rotated, integration is along 2 direction
       	    a[i,j,2] = u[i,j,2] - la[i,j]
            a[i,j,3] = u[i,j,2] - ls[i,j]
            a[i,j,4] = u[i,j,2]
       	    a[i,j,5] = u[i,j,2] + ls[i,j]
       	    a[i,j,6] = u[i,j,2] + la[i,j]
   	        a[i,j,7] = u[i,j,2] + lf[i,j]
        end
   end

   return a #  J&W 2.27+1
end

function interpolateB_center2face(q::Array{Float64,3})

    bxb = SharedArray{Float64}(Nx, Ny) * 0 # needed on 2:N-2 only
    byb = SharedArray{Float64}(Nx, Ny) * 0

    @inbounds for j = 2:Ny-2 # fourth order interpolation
        @inbounds @simd for i = 2:Nx-2
           bxb[i,j] = (-q[i-1,j,5] + 9*q[i,j,5] + 9*q[i+1,j,5] - q[i+2,j,5])/16
           byb[i,j] = (-q[i,j-1,6] + 9*q[i,j,6] + 9*q[i,j+1,6] - q[i,j+2,6])/16
        end
    end

    return bxb, byb
end

function boundaries!(q::Array{Float64,3}, fsy::Array{Float64,2},gsx::Array{Float64,2})

    @inbounds for j = 1:Ny
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

    fsy[1,:]  = fsy[2,:]    = fsy[3,:]
    fsy[:,1]  = fsy[:,2]    = fsy[:,3]
    fsy[Nx,:] = fsy[Nx-1,:] = fsy[Nx-2,:]
    fsy[:,Ny] = fsy[:,Ny-1] = fsy[:,Ny-2]

    gsx[1,:]  = gsx[2,:]    = gsx[3,:]
    gsx[:,1]  = gsx[:,2]    = gsx[:,3]
    gsx[Nx,:] = gsx[Nx-1,:] = gsx[Nx-2,:]
    gsx[:,Ny] = gsx[:,Ny-1] = gsx[:,Ny-2]


   #for i = 2:Nx-1
   #    fsy[i,NBnd-1] = fsy[i,NBnd]
   #    fsy[i,Ny-NBnd+1] = fsy[i,Ny]
   #end

   #@inbounds @simd for i = 2:Nx-1
   #    gsx[i,2] = gsx[i,NBnd]
   #    gsx[i,Ny-NBnd+1] = gsx[i,Ny]
   #end

   fsy[1,:] = fsy[:,1] = 0
   gsx[1,:] = gsx[:,1] = 0

   fsy[Nx,:] = fsy[:,Ny] = 0
   gsx[Nx,:] = gsx[:,Ny] = 0

    return
end

function ES_Switch(q_E::Array{Float64,3}, q_S::Array{Float64,3}, fsy_S::Array{Float64,2}, 
                   gsx_S::Array{Float64,2}, fsy_E::Array{Float64,2}, gsx_E::Array{Float64,2})
	
	q_SE = copy(q_E)
    q_SE[:,:,8] = E2S(q_E) 		# convert q(E) to q(S(E)) so we can compare the 8th component

	dS = abs.(q_SE[:,:,8] - q_S[:,:,8])
	
    bad = find(dS .>= 1e-5)
    
    q = Array{Float64}(Nx,Ny, 9) .* 0
    q[:,:,1:8] = copy(q_S)  	# use q(S) by default

    fsy = copy(fsy_S)
    gsx = copy(gsx_S)
    
    @inbounds for j = 2:Ny
        @inbounds for i = 2:Nx
            if dS[i,j] >= 1e-5
                q[i,j,1:8] = q_SE[i,j,1:8]

                fsy[i,j] = fsy_E[i,j]
                fsy[i-1,j] = fsy_E[i-1,j]
                    
                gsx[i,j] = gsx_E[i,j]       
                gsx[i,j-1] = gsx_E[i,j-1] # need neighbouring component as well
           end
        end
    end

    q[:,:,9] = S2E(q) # save E(S) in the 9th component

	return q, fsy, gsx, bad
end

function E2S(qE::Array{Float64,3})

	E = qE[:,:,8]

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

function weno5_interpolation(q::Array{Float64,3}, a::Array{Float64,3},
							 F::Array{Float64,3}, L::Array{Float64,4},
							 R::Array{Float64,4})
    const Nqx = size(q,1)
    const Nqy = size(q,2)
 
    Fs = zeros(Float64, Nqx, Nqy, 7)

    Fsk = zeros(Float64, 6)
    qsk = zeros(Float64, 6)
    dFsk = zeros(Float64, 5)
    dqsk = zeros(Float64, 5)

	dF = zeros(Float64, Nqx, Nqy, 7)

    @inbounds for m = 1:7
	    @inbounds for j = NBnd:Nqy
	        @inbounds @simd for i = NBnd:Nqx

    			@inbounds for ks = 1:6 # stencil i-2 -> i+3 : 5th order

    				Fsk[ks] = L[i,j,m,1]*F[i-3+ks,j,1] + L[i,j,m,2]*F[i-3+ks,j,2] +
                         	  L[i,j,m,3]*F[i-3+ks,j,3] + L[i,j,m,4]*F[i-3+ks,j,4] +
                         	  L[i,j,m,5]*F[i-3+ks,j,5] + L[i,j,m,6]*F[i-3+ks,j,6] +
                         	  L[i,j,m,7]*F[i-3+ks,j,7]                     
                                                                   
                	qsk[ks] = L[i,j,m,1]*q[i-3+ks,j,1] + L[i,j,m,2]*q[i-3+ks,j,2] +
                         	  L[i,j,m,3]*q[i-3+ks,j,3] + L[i,j,m,4]*q[i-3+ks,j,4] +
                         	  L[i,j,m,5]*q[i-3+ks,j,6] + L[i,j,m,6]*q[i-3+ks,j,7] +
                         	  L[i,j,m,7]*q[i-3+ks,j,8]
    			end # ks

    			first = (-Fsk[2]+7.0*Fsk[3]+7.0*Fsk[4]-Fsk[5]) / 12.0 # J&W eq 2.11

    			for ks = 1:5
    				dFsk[ks] = Fsk[ks+1] - Fsk[ks]
    				dqsk[ks] = qsk[ks+1] - qsk[ks]
    			end

    			amax = max(abs(a[i,j,m]),abs(a[i+1,j,m])) # J&W eq 2.10

    			aterm = 0.5 * (dFsk[1] + amax*dqsk[1]) # Lax-Friedrichs J&W eq. 2.10 & 2.16+
                bterm = 0.5 * (dFsk[2] + amax*dqsk[2])
                cterm = 0.5 * (dFsk[3] + amax*dqsk[3])
                dterm = 0.5 * (dFsk[4] + amax*dqsk[4])

                IS0 = 13.0*(aterm-bterm)^2 + 3.0*(aterm-3.0*bterm)^2
                IS1 = 13.0*(bterm-cterm)^2 + 3.0*(bterm+cterm)^2
                IS2 = 13.0*(cterm-dterm)^2 + 3.0*(3.0*cterm-dterm)^2

                alpha0 = 1.0/(eps+IS0)^2
                alpha1 = 6.0/(eps+IS1)^2
                alpha2 = 3.0/(eps+IS2)^2

                omega0 = alpha0/(alpha0+alpha1+alpha2)
                omega2 = alpha2/(alpha0+alpha1+alpha2)

                second = omega0*(aterm - 2*bterm + cterm)/3 + # phi_N(f+), J&W eq 2.3 + 1
                  	    (omega2-0.5)*(bterm - 2*cterm + dterm)/6

                aterm = 0.5 * (dFsk[5] - amax*dqsk[5]) # Lax-Friedrichs J&W eq. 2.10 & 2.16+
                bterm = 0.5 * (dFsk[4] - amax*dqsk[4])
                cterm = 0.5 * (dFsk[3] - amax*dqsk[3])  
                dterm = 0.5 * (dFsk[2] - amax*dqsk[2])

                IS0 = 13.0*(aterm-bterm)^2 + 3.0*(aterm - 3.0*bterm)^2
                IS1 = 13.0*(bterm-cterm)^2 + 3.0*(bterm + cterm)^2
                IS2 = 13.0*(cterm-dterm)^2 + 3.0*(3.0*cterm - dterm)^2

                alpha0 = 1.0/(eps + IS0)^2
                alpha1 = 6.0/(eps + IS1)^2
                alpha2 = 3.0/(eps + IS2)^2

                omega0 = alpha0/(alpha0 + alpha1 + alpha2)
                omega2 = alpha2/(alpha0 + alpha1 + alpha2)

                third  = omega0*(aterm - 2.0*bterm + cterm) / 3.0 +	# phi_N(f-)
                        (omega2 - 0.5)*(bterm - 2.0*cterm + dterm) / 6.0

                Fs[i,j,m] = first - second + third # J&W eq. 2.16 + 1

		    end # i
	    end # j
	end # m

    @inbounds for m = 1:7
	    @inbounds for j = NBnd:Nqy
	        @inbounds @simd for i = NBnd:Nqx
                dF[i,j,m] = Fs[i,j,1]*R[i,j,m,1] + Fs[i,j,2]*R[i,j,m,2] + # J&W eq. 2.17
        	                Fs[i,j,3]*R[i,j,m,3] + Fs[i,j,4]*R[i,j,m,4] +
                            Fs[i,j,5]*R[i,j,m,5] + Fs[i,j,6]*R[i,j,m,6] +
                            Fs[i,j,7]*R[i,j,m,7] 
            end
        end
     end

	return dF
end # weno5_interpolation()

function rotate_state(q::Array{Float64,3}, direction::String)

    p = Array{Float64}(size(q,2),size(q,1),8)

    if direction == "fwd"

        p[:,:,1] = transpose(q[:,:,1]) # rho
        p[:,:,2] = transpose(q[:,:,3]) # Mx -> Mz
        p[:,:,3] = transpose(q[:,:,4]) # My -> Mx
        p[:,:,4] = transpose(q[:,:,2]) # Mz -> My
        p[:,:,5] = transpose(q[:,:,6]) # Bx -> Bz
        p[:,:,6] = transpose(q[:,:,7]) # By -> Bx
        p[:,:,7] = transpose(q[:,:,5]) # Bz -> By
        p[:,:,8] = transpose(q[:,:,8]) # S or E

    elseif direction == "bwd"

        p[:,:,1] = transpose(q[:,:,1]) # rho
        p[:,:,3] = transpose(q[:,:,2]) # Mz -> Mx
        p[:,:,4] = transpose(q[:,:,3]) # Mx -> My
        p[:,:,2] = transpose(q[:,:,4]) # My -> Mx
        p[:,:,6] = transpose(q[:,:,5]) # Bz -> Bx
        p[:,:,7] = transpose(q[:,:,6]) # Bx -> By
        p[:,:,5] = transpose(q[:,:,7]) # By -> Bz
        p[:,:,8] = transpose(q[:,:,8]) # S or E
    else
        @assert false "Direction parameter : 'fwd' or 'bwd' "
    end

    return p
end # rotate


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

    @inbounds for j = jMin:jMax # Jiang & Wu after eq 2.24
		@inbounds for i = iMin:iMax
			
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

function compute_divB(bxb::Array{Float64,2}, byb::Array{Float64,2})

    divB = similar(bxb) * 0

    @inbounds for j=NBnd:Ny-NBnd
        @inbounds @simd for i=NBnd:Nx-NBnd
        
            divB[i,j] += (bxb[i,j]-bxb[i-1,j])/dx + (byb[i,j]-byb[i,j-1])/dy

        end
    end

    return divB
end

function initial_conditions() # Shock Cloud interaction

	const R0 = 0.12
	const R1 = 0.19

    q = Array{Float64}(Nx, Ny, 9)*0

	# state variables left & right
	#        rho      vx    vy vz Bx        By      Bz    Press
	sl = [2.11820, 13.69836, 0, 0, 0, 19.33647, 19.33647, 65.88816]
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
                fsy::Array{Float64,2}, gsx::Array{Float64,2}, ID::String)

    xidx = [1,2,3,NBnd+1,78,79,80,Nx/2,Nx-NBnd,Nx/2-1,Nx/2,Nx/2+1,NBnd+1,Nx/2,Nx-NBnd, 10,10,10]
    yidx = [1,2,3,NBnd+1,35,35,35,NBnd+1,NBnd+1,Ny/2,Ny/2,Ny/2,Ny-NBnd,Ny-NBnd,Ny-NBnd, 67,68,69]

    @printf "%s \n" ID

    for i=1:length(q)
        if abs(q[i]) < 1e-12
            q[i] = 0
        end
    end
        
   #for i=1:length(bxb)
   #    if abs(bxb[i]) < 1e-12
   #        bxb[i] = 0
   #    end
   #    if abs(byb[i]) < 1e-12
   #        byb[i] = 0
   #    end
   #    if abs(fsy[i]) < 1e-12
   #        fsy[i] = 0
   #    end
   #    if abs(gsx[i]) < 1e-12
   #        gsx[i] = 0
   #    end
   #end

    for i=1:length(xidx)
        
        x = Int64(xidx[i])
        y = Int64(yidx[i])

        @printf "%3i %3i " x-NBnd y-NBnd 
        @printf "%8.3f%8.3f%8.3f%8.3f" q[x,y,1] q[x,y,2] q[x,y,3] q[x,y,4] 
        @printf "%8.3f%8.3f%8.3f%8.3f" q[x,y,5] q[x,y,6] q[x,y,7] q[x,y,8] 
        @printf "%8.3f%8.3f%8.3f%8.3f\n" bxb[x,y] byb[x,y] fsy[x,y] gsx[x,y]
    end
    
    return
end

function printu(u::Array{Float64,2}; ID="")

    const n = size(u, 1)

    xidx = [NBnd+1,floor((n-2*NBnd)/4)+NBnd, n/2-1,n/2,n/2+1,
            floor(3*(n-2*NBnd)/4+NBnd),n-NBnd]

    @printf "%s \n" ID

    for i=1:length(xidx)
        
        x = Int64(xidx[i])

        @printf "%3i  " x-NBnd 
        @printf "%6.4g%6.4g%6.4g%6.4g" u[x,1] u[x,2] u[x,3] u[x,4] 
        @printf "%9.4g%9.4g%9.4g%9.4g\n" u[x,5] u[x,6] u[x,7] u[x,8] 
    end

    return
end

#end # module
