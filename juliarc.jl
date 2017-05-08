# Julia startup file - all line are executed on the startup task

@everywhere push!(LOAD_PATH, "/Users/jdonnert/Dev/src/git/Tandav/lib/julia/")
@everywhere push!(LOAD_PATH, "/Users/jdonnert/Dev/src/git/julia/")
@everywhere push!(LOAD_PATH, "/Users/jdonnert/Dev/src/git/julia/common")

#using ClobberingReload

using CGSUnits 			# Physical constants and conversion
using StringConversion	# convert integer to strings with zero padding
using ArrayStatistics
using Binning
#using Cosmology

# PGF Plotting
using ColorPGF
InitPGF()

# Plots interface, pyplot, modify standard appearance 
import ColorBrewer
colorBrewerSet1 = ColorBrewer.palette("Set1", 9)

using Plots
gr(size=(1024, 1024/sqrt(2)), grid=false, legend=:none, linewidth=1, markersize=1,
	 margin=5mm, color_palette=colorBrewerSet1, xlabel="x", ylabel="y",
 	 markerstrokewidth=0, tickfont=font(7, "Helvetica"), grid=true ) 


# Gadget/Tandav reading

using Tandav

# Wombat I/O

#importall Wombat
