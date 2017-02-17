"""
	Constants and conversions for the CGSM system.
"""
module CGSUnits

export Rgas, Msol, Lsol, me, mec2, qe, mp, grav, c, k_boltz, hq, h_planck,
		sigmat, sigmat_mbarn, sigmapp_mbarn, Tcmb, fine_struct, Jy2cgs, Gyr2s, yr2s,
		mpc2cm, kpc2cm, eV2erg, keV2erg, keV2K, arcmin2rad, H100

#[cgs]
const Rgas	   			= 8.314472e+7 		#[ erg/(K*mol) ]
const Msol				= 1.989e33 		    #[ g ]
const Lsol				= 3.846e+27		    #[ erg/s ]
const me     			= 9.109382e-28		#[ g ]
const mec2		 		= 0.51092111e9	    #[ eV ]
const qe     			= 4.80325e-10		#[ esu ]
const mp     			= 1.6726231e-24	    #[ g ]
const grav     			= 6.672e-8			#[ cm/s^2 ]
const c      			= 2.99792458e+10	#[ cm/s ]
const k_boltz 			= 1.380658e-16		#[ erg/K ]  
const hq     			= 1.054571628e-27	#[ erg * s]
const h_planck			= 6.6260754e-27	    #[ erg * s]
const sigmat 			= 6.653e-25			#[ cm^2 ]
const sigmat_mbarn 		= 665.3				#[ mbarn ]
const sigmapp_mbarn		= 32.0				#[ mbarn ]
const Tcmb	 			= 2.725 		    #[ K ]
const fine_struct     	= 7.29735257e-3     #dimensionless

# conversion to cgs
const Jy2cgs 			= 1.0e-23			#[ erg / s / Hz / cm^2 ]
const yr2s				= 31556926	        #[ s ]
const Gyr2s	 			= 1e9*yr2s       	#[ s ]
const kpc2cm			= 3.08568025e21     #[ cm ]
const mpc2cm			= kpc2cm*1e3	    #[ cm ]
const eV2erg			= 1.60217733e-12	#[ erg/eV ]
const keV2erg			= eV2erg*1e3 		#[ erg/eV ]
const keV2K           	= 1.16e7            #[ K/keV ], ideal gas only
const arcmin2rad 		= 2*pi/360/60		#[ rad ]
const H100 				= 100 * 1e5/mpc2cm 	#[ 1/s ] 100 km/s/Mpc


end
