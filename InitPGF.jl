module InitPGF

importall PGFPlots
import ColorPGF

function initPGF()

	pushPGFPlotsPreamble("\\pgfplotsset{width=10.5cm, height=7.4cm,
					 every axis legend/.append style={draw=none},
					 every axis/.append style={thick, grid=major, tick
					 style={thick}, grid style={very thin}},
					 tick label style={font=\\large},
   					 label style={font=\\large},
   	 				 legend style={font=\\small}}")

	# add this to the string for Sans Serif fonts
	#\\renewcommand{\\familydefault}{\\sfdefault}
	#					 \\usepackage[cm]{sfmath}

	ColorPGF.Set("ALL") # add Brewer colors to PGF: Greens11-GreeNs19 ...

	return
end

end
