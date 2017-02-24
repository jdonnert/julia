# Julia startup file

@everywhere push!(LOAD_PATH, "/Users/jdonnert/Dev/src/git/Tandav/lib/julia/")
@everywhere push!(LOAD_PATH, "/Users/jdonnert/Dev/src/git/julia/")
@everywhere push!(LOAD_PATH, "/Users/jdonnert/Dev/src/git/julia/common")

#using ClobberingReload

using CGSUnits 	# Physical constants and conversion
using ArrayStatistics
using Binning
using Cosmology

# PGF Plotting
## using ColorPGF
### InitPGF()

# Plots interface, pyplot, modify standard appearance 
import ColorBrewer
colorBrewerSet1 = ColorBrewer.palette("Set1", 9)

using Plots

pyplot(size=(640,480), grid=false, legend=:none, linewidth=1, markersize=1,
	   left_margin=[5mm 15mm], bottom_margin=[5mm 15mm], 
	   color_palette=colorBrewerSet1) 


# Gadget/Tandav reading

using Tandav
tandav = TandavCodeObject()

# Wombat I/O

#importall Wombat
#wb = WombatCodeObject()
