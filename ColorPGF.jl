"""
We make the Brewer Colors available to PGFPlots Plot.Linear.
the colors are available as Br+number: 
p = Plots.Linear(x,y, style="Br4")
"""
module ColorPGF

using PGFPlots
using ColorTypes

export Set, InitPGF

const seq_names = [ "OrRd", "PuBu", "BuPu", "Oranges", "BuGn", 
		 			"YlOrBr", "YlGn", "Reds", "RdPu", "Greens", 
					"YlGnBu", "Purples", "GnBu", "Greys", "YlOrRd",
					"PuRd", "Blues", "PuBuGn" ]
const seq_nCol = fill(9, size(seq_names))

const div_names = [ "Spectral", "RdYlGn", "RdBu", "PiYG", "PRGn", 
					"RdYlBu", "BrBG", "RdGy", "PuOr" ]
const div_nCol = fill(11, size(div_names))

const qual_names = ["Set1", "Set2", "Set3", "Accent", "Dark2", 
		  	 		"Paired", "Pastel1", "Pastel2" ]
const qual_nCol = [9,8,12,8,8,12,9,8]

function InitPGF()

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

function Set(name::AbstractString)

	if name != "ALL"
	
		nCol = get_nCol(name)

		@assert(nCol > 0, "Color Table $name not found")

		println("Setting <$nCol> colors in Brewer Palette '$(name)'.")

		colors = ColorBrewer.palette(name, nCol)
	
		for i = 1:nCol 

			ct_name = "Br"*string(i)

			r = ColorTypes.red(colors[i,1])
			g = ColorTypes.green(colors[i,1])
			b = ColorTypes.blue(colors[i,1])

			rgb = floor(Int16, [r,g,b]*255) # convert FixedPointFloat to Integer
			
			PGFPlots.define_color(ct_name, rgb) # set nCol LaTeX colors
		end
	
	else # set them all!!

		for j=1:length(seq_names)

			nCol = get_nCol(seq_names[j])

			colors = ColorBrewer.palette(seq_names[j], nCol)

			for i=1:nCol
		
				ct_name = seq_names[j]*string(i)

				r = ColorTypes.red(colors[i,1])
				g = ColorTypes.green(colors[i,1])
				b = ColorTypes.blue(colors[i,1])

				rgb = floor(Int16, [r,g,b]*255)
			
				PGFPlots.define_color(ct_name, rgb)
			end
		end

		for j=1:length(div_names)

			nCol = get_nCol(div_names[j])
			
			colors = ColorBrewer.palette(div_names[j], nCol)

			for i=1:nCol
		
				ct_name = div_names[j]*string(i)
		
				r = ColorTypes.red(colors[i,1])
				g = ColorTypes.green(colors[i,1])
				b = ColorTypes.blue(colors[i,1])

				rgb = floor(Int16, [r,g,b]*255)
			
				PGFPlots.define_color(ct_name, rgb)
			end
		end

		for j=1:length(qual_names)

			nCol = get_nCol(qual_names[j])

			colors = ColorBrewer.palette(qual_names[j], nCol)

			for i=1:nCol
		
				ct_name = qual_names[j]*string(i)
		
				r = ColorTypes.red(colors[i,1])
				g = ColorTypes.green(colors[i,1])
				b = ColorTypes.blue(colors[i,1])

				rgb = floor(Int16, [r,g,b]*255)
			
				PGFPlots.define_color(ct_name, rgb)
			end
		end
	end

	return
end

function get_nCol(name::AbstractString)

	tables = cat(1, seq_names, div_names, qual_names)
	
	nCol = vcat(seq_nCol, div_nCol, qual_nCol)

	for i = 1:size(tables,1)

		if name == tables[i]
	
			return nCol[i]

		end

	end # i

	return 0
end

end # brewer_tables
