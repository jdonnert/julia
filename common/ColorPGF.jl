# We define Brewer Colors to use with PGFPlots.
# the colors are available as numbers: 
# p = Plots.Linear(x,y, style="Br1")

module ColorPGF

using ColorBrewer
using PGFPlots

export Set

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

function Set(name::AbstractString)

	nCol = get_num_colors(name)

	@assert(nCol > 0, "Color Table $name not found")

	colors = ColorBrewer.palette(name, nCol)

	for i = 1:nCol 

		ct_name = "Br"*string(i)
		PGFPlots.define_color(ct_name, colors[i,1]) # set nCol LaTeX colors
	end

	return
end

function get_num_colors(name::AbstractString)

	tables = cat(1, seq_names, div_names, qual_names)
	
	nCol = vcat(seq_nCol, div_nCol, qual_nCol)

	for i = 1:size(tables,1)

		if name == tables[i]
	
			println("Found <$(nCol[i])> colors in Brewer Palette '$(name)'.")

			return nCol[i]

		end

	end # i

	return 0
end

end # brewer_tables
