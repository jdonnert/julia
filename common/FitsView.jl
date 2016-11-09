module FitsView

using FITSIO
using Colors
using ColorTypes
using FixedPointNumbers
using PyPlot
#using PerceptualColourMaps

export Fits_View

function Fits_View(input; fout="./fview.pdf", scale=0., ext=1, log=false,
				   slice=1, zmin=0., zmax=0., unitName="", unitScale=1.,
				   scaleText=" ", scaleCol="black", annoText=" ", annoCol="black", 
				   movie=false, colmap=1,  pythonCMaps=false, noColBar=false)
	
	if isa(input, String) # make sure its an array
		input = [input]
	end

	if isa(input[1], String)
		nImg = length(input)
	else
		nImg = length(slice)
	end

	println("Found $nImg images")
	
	slice = arrayfy(slice, nImg) # make parameters indexable by nImg
	unitScale = arrayfy(unitScale, nImg)
	zmin = arrayfy(zmin, nImg)
	zmax = arrayfy(zmax, nImg)
	annoText = arrayfy(annoText, nImg)
	colmap = arrayfy(colmap, nImg)
	annoCol = arrayfy(annoCol, nImg)
	scaleCol = arrayfy(annoCol, nImg)
	
	pygui(false) # make figure
	
	plt[:rc]("font", family="serif")
	plt[:rc]("font", size="12")

	nxFig = 4
	nyFig = 5

	if noColBar == true
		nyFig = nxFig
	end

	if movie == false
		fig = figure(figsize=[nxFig,nyFig], dpi=600, tight_layout=true)
	end

	zrange = [0,0.]

	for i = 1:nImg

		if movie == true
			fig = figure(figsize=[nxFig,nyFig], dpi=600, tight_layout=true)
		end

		extend = find_partition(i, nImg, nxFig, nyFig)

		fig[:add_axes](extend)
	
		ax = gca()

		ax[:set_xlim]([0,1024])
		ax[:set_ylim]([0,1024])

		ax[:set_xlabel](" ")
		ax[:set_ylabel](" ")
	
		ax[:set_xticklabels](" ")
		ax[:set_yticklabels](" ")

		minorticks_on()

		tick_params(width=0.5, length=3, which="major")
		tick_params(width=0.3, length=2, which="minor")

		xticks(Array{Float64}(linspace(0,1024,5)))
		yticks(Array{Float64}(linspace(0,1024,5)))

		if typeof(input[i]) == String

			img = (FITS(input[i]))[ext]
			
			img = img[:,:, slice[i]]
		else

			img = input[:,:, slice[i]]
		end

		img = transpose(img)

		img *= unitScale[i]
		
		if zmin[i] == zmax[i]

			zmin[i] = minimum(img)
			zmax[i] = maximum(img)
			
			println("zrange of image <$i> = $(zmin[i]), $(zmax[i])")
		end

		zrange[1] = zmin[i]
		zrange[2] = zmax[i]
			
		bad = find(img .< zrange[1])
		img[bad] = zrange[1]

		bad = find(img .> zrange[2])
		img[bad] = zrange[2]

		if pythonCMaps == false
			
			#map = cmap(colmap[i])
			#map = ColorMap(map[1])
			map = mycmap(colmap[i])
		else
			map = get_cmap(colmap[i])
		end
			

		if log == true

			imshow(log10(img), interpolation="none", cmap=map)
		else

			imshow(img, interpolation="none", cmap=map)
		end

		# annotations 
		
		if (i == 1) && scaleText != " "

			text = latexstring("\\rm "*annoText[i])
			x0 = extend[1] + 0.65*extend[3]
			y0 = extend[2] + 0.05*extend[4]
			
			fig[:text](x0, y0, scaleText, color=scaleCol[i], size="xx-small")
		end

		if  annoText[i] != " "

			text = latexstring("\\rm "*annoText[i])
			x0 = extend[1] + 0.05*extend[3]
			y0 = extend[2] + 0.9*extend[4]

			fig[:text](x0, y0, text, color=annoCol[i], size="xx-small")

		end

		make_contours()

		if (movie == true) && (noColBar == false)
				
			make_colbar(fig, zrange, colmap[i]; unitName, log)

			savefig(fout*"_$i.jpg")

			close(fig)
		end

	end
	
	if (movie == false) && (noColBar == false)

		make_colbar(fig, zrange, colmap[1], name=unitName, log=log,
			  			pythonCMaps=pythonCMaps)
	end

	savefig(fout)

	close(fig)

	return
end

function make_colbar(fig, zrange, colmap; name=" ", log=false, pythonCMaps=false)

	const N = 1024
	
	fig[:add_axes]([0.1, 0.12, 0.8, 0.06])

	cax = gca()
	
	# y ticks 

	cax[:set_ylim]([1,75])
	cax[:set_yticklabels](" ")
	yticks([0,75])

	cax[:set_xlim](0, N-1)

	cbar = Array{Float64}(1024, N)

	for i = 1:1024
		for j = 1:N
			cbar[i,j] = j
		end
	end

	# x ticks

	delta = 0

	if log == true 

		@assert(zrange[1] != 0, "zrange cannot be 0 in log scale")

		zrange = log10(zrange)
		
		nTicks = Integer(round(abs(floor(zrange[2]) - floor(zrange[1]))))
		
		if zrange[1] == floor(zrange[1]) && zrange[2] == floor(zrange[2])
			nTicks += 1 # both ends
		end
		
		i = linspace(1,nTicks, nTicks)

		mTickVal = i .* (zrange[2]-zrange[1])/nTicks + zrange[1] 
		mTickVal = floor(mTickVal)
		
		val2tick = N / (zrange[2] - zrange[1])

		mTicks = Array{Float64}((mTickVal - zrange[1]) * val2tick)

	else
		delta = 10.0^Integer(round(log10(zrange[2] - zrange[1])) - 1)
		
		minTick = ceil(zrange[1]/delta) * delta
		maxTick = floor(zrange[2]/delta) * delta

		nTicks = Integer((maxTick - minTick) / delta) + 1

		mTickVal = linspace(minTick, maxTick, nTicks)
		mTickVal = Array{Float64}(mTickVal)

		val2tick = N / (zrange[2] - zrange[1])

		mTicks = Array{Float64}(mTickVal - zrange[1]) * val2tick
		
	end

	cax[:set_xticks](mTicks)

	# labels 

	xlabels = Array(LaTeXString, nTicks)

	for i=1:nTicks

		x = mTickVal[i]

		if log == true

			x = 10^x
		end

		if x == 0

			xlabels[i] = latexstring("0")

			continue
		end
		
		if nTicks > 100 && i%50 != 0
			
			xlabels[i] = LaTeXString("")

			continue
		end

		if nTicks > 15 && i%5 != 0
			
			xlabels[i] = LaTeXString("")

			continue
		end

		if (delta > 0.01) && (delta < 1000) && (log == false)

			mant = @sprintf("%3.0f", x)
			xlabels[i] = latexstring(mant)
		else

			expo = @sprintf("%i", Integer(floor(log10(x))))
			
			mant = x/10^floor(log10(x))
	
			if mant == 1.0
				xlabels[i] = latexstring("10^{"*expo*"}")
			else
				mant = @sprintf("%2.1f", mant)
				xlabels[i] = latexstring(mant*"\\times10^{"*expo*"}")
			end
		end
	end
	
	cax[:set_xticklabels](xlabels)

	#minors

	dMajor = mTickVal[2] - mTickVal[1]
	bounds = [mTickVal[1]-dMajor; mTickVal; mTickVal[nTicks]+dMajor]
	
	minor = []

	if log == true

		bounds = 10.^bounds
		zrange = 10.^zrange
	end

	for i=1:nTicks+1

		for j=2:9
			
			val = j*bounds[i]   
			
			if val < zrange[1]
				continue
			end

			if val > zrange[2]
				break
			end

			minor = [minor; val ]
		end
	end

	minor = Array{Float64}(minor)

	if log == true
		minor = log10(minor)
		zrange = log10(zrange)
	end

	minor = (minor .- zrange[1]) .* val2tick

	cax[:set_xticks](minor, minor=true)

	if pythonCMaps == false
			
		#map = cmap(colmap) # see PerceptualColourMaps documentation
		#map = ColorMap(map[1])
		map = mycmap(colmap)
	else
		map = get_cmap(colmap)
	end

	imshow(cbar, interpolation="hanning", cmap=map)

	xlabel(name)
end

function make_contours()

end

function arrayfy(input, nImg)

	if isa(input, String)
		input = [input]
	end

	if length(input) != nImg
	
		output = input

		for i=2:nImg
			output = [output; input]
		end

	elseif ! isa(input, Array)
		output = [input]
	else
		output = input
	end

	return output
end

function find_partition(i, nImg, nx, ny) #

	x0 = 0.005 # just to give a little bit of room on the sides
	y0 = 1 - (nx - 2*x0)/ny

	nxy = ceil(sqrt(nImg))

	xsize = (1-2*x0)/nxy # square images
	ysize = (1-y0)/nxy # square images

	dx = ((i-1) % nxy) * xsize
	dy = ((nxy-1) - floor((i-1)/nxy)) * ysize

	extend = [x0+dx, y0+dy-x0/2, xsize, ysize]

	if i==3 && nImg==3

		extend[1] += 0.5*extend[3]
	end
	
	return extend
end

function mycmap(i) # http://peterkovesi.com/projects/colourmaps/index.html

	path = "/Users/jdonnert/Dev/src/git/julia/common/colmaps/"

	fnames = [
	"_bgyr_35-85_c73_n256.csv"	,					# 1
   	"cyclic_grey_15-85_c0_n256.csv",				# 2
   	"cyclic_grey_15-85_c0_n256_s25.csv",
    "cyclic_mrybm_35-75_c68_n256.csv",
    "cyclic_mrybm_35-75_c68_n256_s25.csv",
    "cyclic_mygbm_30-95_c78_n256.csv",
    "cyclic_mygbm_30-95_c78_n256_s25.csv",
	"cyclic_wrwbw_40-90_c42_n256.csv",
    "cyclic_wrwbw_40-90_c42_n256_s25.csv",
    "diverging-isoluminant_cjm_75_c23_n256.csv",	# 10
    "diverging-isoluminant_cjm_75_c24_n256.csv",
    "diverging-isoluminant_cjo_70_c25_n256.csv",
    "diverging-linear_bjr_30-55_c53_n256.csv",
    "diverging-linear_bjy_30-90_c45_n256.csv",
	"diverging-rainbow_bgymr_45-85_c67_n256.csv",
	"diverging_bkr_55-10_c35_n256.csv",
    "diverging_bky_60-10_c30_n256.csv",
    "diverging_bwr_40-95_c42_n256.csv",
    "diverging_bwr_55-98_c37_n256.csv",
    "diverging_cwm_80-100_c22_n256.csv",
    "diverging_gkr_60-10_c40_n256.csv",
    "diverging_gwr_55-95_c38_n256.csv",
    "diverging_gwv_55-95_c39_n256.csv",				# 23
    "isoluminant_cgo_70_c39_n256.csv",
    "isoluminant_cgo_80_c38_n256.csv",
    "isoluminant_cm_70_c39_n256.csv",				# 26
    "linear_bgy_10-95_c74_n256.csv",
    "linear_bgyw_15-100_c67_n256.csv",
    "linear_bgyw_15-100_c68_n256.csv",
    "linear_blue_5-95_c73_n256.csv",
    "linear_blue_95-50_c20_n256.csv",
    "linear_bmw_5-95_c86_n256.csv",
    "linear_bmw_5-95_c89_n256.csv",
    "linear_bmy_10-95_c71_n256.csv",
    "linear_bmy_10-95_c78_n256.csv",
    "linear_gow_60-85_c27_n256.csv",
    "linear_gow_65-90_c35_n256.csv",
    "linear_green_5-95_c69_n256.csv",
    "linear_grey_0-100_c0_n256.csv",
    "linear_grey_10-95_c0_n256.csv",
    "linear_kry_5-95_c72_n256.csv",
    "linear_kry_5-98_c75_n256.csv",
    "linear_kryw_5-100_c64_n256.csv",
    "linear_kryw_5-100_c67_n256.csv",
    "linear_ternary-blue_0-44_c57_n256.csv",
    "linear_ternary-green_0-46_c42_n256.csv",
	"linear_ternary-red_0-50_c52_n256.csv",	
	"rainbow_bgyr_35-85_c72_n256.csv",		# 48
	"rainbow_bgyr_35-85_c73_n256.csv",
	"rainbow_bgyrm_35-85_c69_n256.csv",
 	"rainbow_bgyrm_35-85_c71_n256.csv" ]			# 51

	fin = path*fnames[i]

	#println("Colormap File $fin")

	tmp = readdlm(fin, ',', Float64) / 255	
	tmp = Array{UFixed8}(tmp)

	N = Integer((length(tmp))/3)

	rgb = Array{RGB{U8}}(N)
	for j=1:N
		rgb[j] = RGB(tmp[j,1],tmp[j,2],tmp[j,3])
	end

	mymap = ColorMap("A", rgb)

	return mymap
end

end # module

