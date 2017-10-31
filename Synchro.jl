"""
Synchrotron Solver for arbitrary electron spectra.
    
    Synchro.j_nu(), Synchro.kernel_F()
"""
module Synchro 

import Dierckx # cubic spline interpolation
import Plots
import ColorBrewer

export j_nu, kernel_F, Test_Synchro #, kernel_G

const me = 9.1093820e-28
const c = 2.9979246e10
const qe = 4.8032500e-10
 
const nu_c_prefac = 3*qe/(4*pi*me^3*c^3) # GS 2.16 for p=E/c
const j_nu_prefac = qe^3*sqrt(3)/(me*c^2) # GS 3.20

const N = 64
const xmax = 21     # edges of the synchro kernel considered
const xmin = 1e-20

const N_a = 64      # pitch angle
const amin = 0 
const amax = pi/2

const x_tab = log10.([1e-4,1e-3,1e-2,3e-2,0.1,0.2,0.28,0.3,0.5,0.8,1,2,3,5,10])
const K_tab = log10.([0.0996,0.213,0.445,0.613,0.818,0.904,0.918,0.918,0.872,
					  0.742,0.655,0.301,0.130,2.14e-2,1.92e-4]) # Longair Tab 8.1

const spl = Dierckx.Spline1D(x_tab, K_tab) # interpolate

"""
Calculate the synchrotron emissivity in erg/s/cm^3/Hz at frequency "nu" 
for a magnetic field "B" at pitch angle "alpha" from a CRe spectrum 
"Ncre(p::Array{Float64,1})" defined over momentum n(p) dp !

    jnu = j_nu(nu::Real, B::Real, alpha::Real, Ncre::Function)

If "alpha" is not defined, calculate the pitch angle averaged 
synchrotron emissivity instead: 

    jnu_avg = j_nu(nu::Real, B::Real, Ncre::Function)

The functions are accurate to about <1% for a power law spectrum. 
The error is dominated by the accuracy of the synchrotron kernel used.

To test the solver, run:

    Test_Synchro()

See Ginzburg & Syrovatskii 1965, "Cosmic Magnetobremsstrahlung"

Please cite: Donnert J. M. F., Stroe A., Brunetti G., Hoang D., Roettgering H.,
             2016, MNRAS, 462, 2014

License: MIT
"""
function j_nu(nu::Real, B::Real, alpha::Real, Ncre::Function)

    sin_a = sin(alpha)

    pmin = sqrt(nu/(nu_c_prefac * B * sin_a * xmax)) # momentum edges
	pmax = sqrt(nu/(nu_c_prefac * B * sin_a * xmin))

    di = log(pmax/pmin)/(N-1)

    p = Array{Float64}(N)
    pmid = Array{Float64}(N)
    dp = Array{Float64}(N)
    
    @inbounds @fastmath @simd for i=1:N
        p[i] = pmin * exp(di*(i-1))
        dp[i] = pmin * (exp(di*(i-1)) - exp(di * (i-2)))
        pmid[i] = 0.5 * pmin * (exp(di*(i-1)) + exp(di*(i-2)))
    end    

    np = Ncre(p)
    npmid = Ncre(pmid)

    # F via syntactic loop fusion
    F = Array{Float64}(N)
    Fmid = Array{Float64}(N)

    @. F = np * kernel_F(nu/(nu_c_prefac * B * sin_a) / p^2)
    @. Fmid = npmid * kernel_F(nu/(nu_c_prefac * B * sin_a) / pmid^2)

    int = 0.0

    @inbounds @fastmath @simd for i=2:N
        int += B * sin_a * dp[i]/6.0 * (F[i] + F[i-1]+4*Fmid[i]) # Simpson rule
    end

    return j_nu_prefac * int # GS, eq. 3.20
end

# with pitch angle integration
function j_nu(nu::Real, B::Real, Ncre::Function)

    da = (amax - amin)/(N_a-1)

    last_pint = 0
    int = 0.0

    for j = 2:N_a # pitch angle integral

        a = amin + (j-1)*da

        pint = j_nu(nu, B, a, Ncre)

        if  last_pint == 0
            last_pint = pint
        end

        int += sin(a)/4/pi * da * (pint+last_pint)/2 # trapezoidal rule
        
        last_pint = pint
    end

    return int
end

"""
Synchrotron kernel function F(x) using the Dierckx package for interpolation

Usage: 

    Kx[:] = kernel_F(x::Array{Float64,1})

    Kx = kernel_F(x::Float64)

See Ginzburg & Syrovatskii 1965, Longair 2011
"""
function kernel_F(x::Array{Float64,1})

    Kx = similar(x)

    @inbounds @fastmath for i=1:size(x,1)
        if x[i] < 1e-4 
            Kx[i] = 2.7082358788528 * (x[i]/2)^(1/3) # approx for small x
        elseif x[i] > 10 
            Kx[i] = 1.2533141373155 * exp(-x[i])*sqrt(x[i]) # approx for large x
        else
            Kx[i] = 10^spl(log10(x[i])) # cubic spline interpolation
        end
    end

    return Kx
end

function kernel_F(x::Float64)

    Kx = 0.0

    if x < 1e-4 
        @fastmath Kx = 2.7082358788528 * (x/2)^(1/3)
    elseif x > 10 
        @fastmath Kx = 1.2533141373155 * exp(-x)*sqrt(x)
    else
        @fastmath Kx = 10^spl(log10(x))
    end

    return Kx
end

function show_synchrotron_kernel()

    Plots.gr(color_palette=ColorBrewer.palette("Set1", 9),legend=false, 
    	     grid=true, markerstrokewidth=0, markersize=2, linewidth=1,
             wsize=(640,640/sqrt(2)))

    N = 256

    x = logspace(-20,3.4,N)

    Kx = kernel_F(x) # Array version

    Kx2 = similar(x)
    for i=1:N
        Kx2[i] = kernel_F(x[i]) # Scalar version
    end

    Plots.plot(x, Kx, xscale=:log10, yscale=:log10, xlims=(1e-8,100), 
         ylims=(1e-4,1), wsize=(1024,768),xlabel="x", ylabel="F(x)")

    Plots.scatter!(x, Kx2, markersize=3)
end

# make parameters global const variables for speed !

const s = 2.1984127 # Sausage relic (Donnert et al. 2016)
const n0 = 1.62e-28
const B = 5e-6

function CReSpectrum(p::Array{Float64,1})

    return n0 * p.^(-s)
end

function Test_Synchro()
 
    alpha = pi/4
    
    nuarr = [1,70,150,320,610,1400,2100,3200, 9e3, 16e3, 32e3] * 1e9 # GHz

    N = size(nuarr,1)
    
    j_num  = Array{Float64}(N)

    println("\nNot pitch angle averaged, s=$s, n0=$n0, B=$B, alpha=$alpha")

    prefac = sqrt(3) * qe^3 / (me * c^2  * (s + 1)) *  # GS, eq. 3.26 
	         gamma(s/4 + 19/12) * gamma(s/4 - 1/12) *
	         ( (3 * qe)/(me^3 * c^3 * 2 * pi))^((s - 1)/2) #  for N(p) dp

    j_GS = n0 * prefac * nuarr.^(-0.5*(s-1)) * (B*sin(alpha))^(0.5*(s+1)) # erg/s/cm^3
        
    for i = 1:N

        nu = nuarr[i]

        j_num[i] = j_nu(nu, B, alpha, CReSpectrum) # no pitch ang integration
    
        @printf("nu = %6d j_ana = %g j_num = %g error = %g\n", 
                nu/1e9, j_GS[i], j_num[i], (j_num[i]-j_GS[i])/j_GS[i])
    end

    println("\nPitch angle averaged, s=$s, n0=$n0, B=$B")

    agam = 2^(s/2-1/2) * sqrt(3)   * gamma(s/4 - 1/12) * gamma(s/4 + 19/12) / 
          (8 * sqrt(pi) * (s + 1)) * gamma((s+5)/4) / gamma((s+7)/4) # GS eq 3.32
	          
    j_GS = agam * qe^3/(me*c^2) * (3 * qe/(me^3 * c^3 * 4 * pi))^((s - 1)/2) * 
           n0 * nuarr.^(-(s-1)/2) * B^((s+1)/2) # GS, eq. 3.31, erg/s/cm^3
        
    for i = 1:N

        nu = nuarr[i]

        j_num[i] = j_nu(nu, B, CReSpectrum) # no pitch ang integration
    
        @printf("nu = %6d j_ana = %g j_num = %g error = %g\n", 
                nu/1e9, j_GS[i], j_num[i], (j_num[i]-j_GS[i])/j_GS[i])
    end

    Nspec = 2^20
    Nthreads = Threads.nthreads()

    println("\nRunning $Nspec spectra, $Nthreads threads: ")

    val = SharedArray{Float64}(Nspec)

    tic()
    
    Threads.@threads for i = 1:Nspec
        val[i] = j_nu(1.4e9, B, pi/2, CReSpectrum)
    end

    toc()

    return
end

end #module Synchro
