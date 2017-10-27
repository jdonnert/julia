module Synchro 

using Dierckx # cubic spline interpolation

const me = 9.1093820e-28
const c = 2.9979246e10
const qe = 4.8032500e-10
 
const nu_c_prefac = 3*qe/(4*pi*me^3*c^3) # GS 2.16 for p=E/c
const j_nu_prefac = qe^3*sqrt(3)/(me*c^2) # GS 3.20

const N = 1024
const N_a = 1

const amin = 0      # angles
const amax = pi/2

const xmax = 21     # edges of the synchro kernel 
const xmin = 1e-20

export j_nu, Kernel_F#, Kernel_G

p = Array{Float64}(N)
pmid = Array{Float64}(N)
dp = Array{Float64}(N)

np = Array{Float64}(N)
npmid = Array{Float64}(N)

x = Array{Float64}(N)
F = Array{Float64}(N)
xmid = Array{Float64}(N)
Fmid = Array{Float64}(N)

"""
Calculate the synchrotron emissivity in erg/s/cm^3 at frequency "nu" 
for a magnetic field "B" at pitch angle "alpha" from a CRe spectrum 
"Ncre(p::Array{Float64,1})" defined over momentum n(p) dp !

    jnu = synchrotron_emissivity(nu::Real, B::Real, alpha::Real, Ncre::Function)

If the alpha is not defined, calculate the pitch angle averaged 
synchrotron emissivity instead: 

    jnu_avg = synchrotron_emissivity(nu::Real, B::Real, Ncre::Function)

The pitch angle averaged emissivity is about a factor 50 more expensive 
than the standard integral.

See Ginzburg & Syrovatskii 1965, "Cosmic Magnetobremsstrahlung"

Please cite: Donnert J. M. F., Stroe A., Brunetti G., Hoang D., Roettgering H.,
2016, MNRAS, 462, 2014
"""
function j_nu(nu::Real, B::Real, alpha::Real, Ncre::Function)

    pmin = sqrt(nu/(nu_c_prefac * B * xmax)) # momentum edges
	pmax = sqrt(nu/(nu_c_prefac * B * xmin))

    di = log(pmax/pmin)/(N-1)

    sin_a = sin(alpha)

    @inbounds @simd for i=1:N
        p[i] = pmin * exp(di*(i-1))
        dp[i] = pmin * (exp(di*(i-1)) - exp(di * (i-2)))
        pmid[i] = 0.5 * pmin * (exp(di*(i-1)) + exp(di*(i-2)))
    end    

    for i=1:N
        np[i] = Ncre(p[i])
        npmid[i] = Ncre(pmid[i])

        x[i] = nu/(nu_c_prefac * B * sin_a) ./ p[i]^2
        xmid[i] = nu/(nu_c_prefac * B * sin_a) ./ pmid[i]^2
    end

    F = np .* kernel_F(x)
    Fmid = npmid .* kernel_F(xmid)

    int = 0.0

    @inbounds @simd for i=2:N
        int += B * sin_a * dp[i]/6.0 * (F[i] + F[i-1]+4*Fmid[i]) # Simpson rule
    end

    return j_nu_prefac * int # GS, eq. 3.20
end

function j_nu(nu::Real, B::Real, Ncre::Function)

    const nu_c_prefac = 3*qe/(4*pi*me^3*c^3)
	const j_nu_prefac = qe^3*sqrt(3)/(me*c^2)

	const N = 4096
	const N_a = 1

    const xmax = 21     # edges of the synchro kernel 
    const xmin = 1e-20

    sina = sin(alpha) # sin of pitch angle

    pmin = sqrt(nu/(nu_c_prefac * B * xmax)) # momentum edges
	pmax = sqrt(nu/(nu_c_prefac * B * xmin))

    di = log(pmax/pmin)/(N-1)

    p = Array{Float64}(N)   # momentum array
    pmid = Array{Float64}(N)
    dp = Array{Float64}(N)

    @inbounds @simd for i=1:N
        p[i] = pmin * exp(di*(i-1))
        dp[i] = pmin * (exp(di*(i-1)) - exp(di * (i-2)))
        pmid[i] = 0.5 * pmin * (exp(di*(i-1)) + exp(di*(i-2)))
    end    

    np = Ncre(p)

    npmid = Ncre(pmid)

    da = (amax - amin)/(N_a-1) # angle

    last_pint = 0
    int = 0.0

    for j = 1:N_a

        sin_a2 = 1 #sin(amin + (j-1)*da)^2

        x = nu/(nu_c_prefac * B * sin_a2) ./ p.^2
        F = np .* kernel_F(x)
    
        xmid = nu/(nu_c_prefac * B*sin_a2) ./ pmid.^2
        Fmid = npmid .* kernel_F(xmid)
        
        pint = 0.0

        for i=2:N
            pint += B * sin_a2 * dp[i]/6.0 * (F[i] + F[i-1]+4*Fmid[i]) # Simpson rule
        end

        if  last_pint == 0
            last_pint = pint
        end

        int += pint #da * 0.5 * (pint+last_pint) # trapezoidal rule
        
        last_pint = pint
    end

    return j_nu_prefac * int
end

"""
Synchrotron kernel function F(x) using the Dierckx package for interpolation

Usage: 
    Kx = kernel_F(x::Array{Float64,1})

This is accurate to about 0.5% for a power law spectrum.

Donnert 2017
"""

function kernel_F(x::Array{Float64,1})

    N = size(x,1)
    Kx = similar(x)

    x_tab = log10.([1e-4,1e-3,1e-2,3e-2,0.1,0.2,0.28,0.3,0.5,0.8,1,2,3,5,10])
	K_tab = log10.([0.0996,0.213,0.445,0.613,0.818,0.904,0.918,0.918,0.872,
					0.742,0.655,0.301,0.130,2.14e-2,1.92e-4]) # Longair Tab 8.1

    spl = Spline1D(x_tab, K_tab)

    @inbounds for i=1:N
        if x[i] < 1e-4 
            Kx[i] = 2.708235878852801 * (x[i]/2)^(1/3) # approx for small x
        elseif x[i] > 10 
            Kx[i] = 1.2533141373155001*exp(-x[i])*sqrt(x[i]) # approx for large x
        else
            Kx[i] = 10^spl(log10(x[i])) # cubic spline interpolation
        end
    end

    return Kx
end

function show_synchrotron_kernel()

    Plots.gr(color_palette=ColorBrewer.palette("Set1", 9),legend=false, 
    	     grid=true, markerstrokewidth=0, markersize=1, linewidth=1,
             wsize=(640,640/sqrt(2)))

    N = 256

    x = logspace(-10,3,N)

    Kx = kernel_F(x)

    scatter(x, Kx, xscale=:log10, yscale=:log10, xlims=(1e-3,10), 
                   ylims=(0.1,1), wsize=(1024,768))
end

# make parameters global const variables

function CReSpectrum(p::Array{Float64,1})

    return n0 * p.^(-s)
end

function test_synchrotron_brightness()
 
    global const s = 2.1984127 # (Donnert et al. 2016)
    global const n0 = 1.62e-28

    B = 5e-6
    
    nuarr = [70,150,320,610,1400,2100,3200, 9000, 16000, 32000] * 1e9 # GHz

    N = size(nuarr,1)
    
    prefac = sqrt(3) * qe^3 / (me * c^2  * (s + 1)) *  # GS, eq. 3.26 but for N(p) dp 
	         gamma(s/4 + 19/12) * gamma(s/4 - 1/12) *
	         ( (3 * qe)/(me^3 * c^3 * 2 * pi))^((s - 1)/2)

    j_GS = n0 * prefac * nuarr.^(-0.5*(s-1)) * B^(0.5*(s+1)) # erg/s/cm^3

    j_num  = Array{Float64}(N)
    
    println("Not pitch angle averaged, s=$s, n0=$n0, B=$B")
        
    for i = 1:N

        nu = nuarr[i]

        j_num[i] = j_nu(nu, B, pi/2, CReSpectrum) # no pitch ang integration
    
        @printf("nu = %6d j_GS = %g j_num = %g error = %g\n", 
                nu/1e9, j_GS[i], j_num[i], (j_num[i]-j_GS[i])/j_GS[i])
    end

    Nspeed = 2^15
    Ncpu = Threads.nthreads()

    println("Running $Nspeed emissions, $Ncpu CPUs: ")

    tic()

    val = Array{Float64}(Nspeed)

    for i = 1:Nspeed
        val[i] = j_nu(1.4e9, B, pi/2, CReSpectrum)
    end

    toc()

    #plot(nuarr/1e9, j_GS, j_num)

    return
end

end #module Synchro
