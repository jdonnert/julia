module Binning

export BinArray

"""
Sort array elements into buckets, multithreaded !
Returns a tuple with the data mean in the bins, the bin positions and the 
number of items in each bin. Bins are set either by number or by position. 
Supports logarithmic binning.

Example 1:

	data = randn(2048)

	bMean, bPos, bCnt = BinArr(zeros(2048), pos=data, nbins=64)

	plot(bPos, binCnt) # plot a Gaussian

Example 2:

	bMean, bPos = BinArr(rho, pos=r, nbins=128, log=true)

	plot(bPos, bMean, xscale=:log10, yscale=:log10) # plot radial profile

"""
function BinArray(arr::Array; pos=-1, bin_low=-1, log=false, nbins=0)

	@assert(ndims(arr) == 1, "\nCan't bin multi-dimensional arrays\n")

	N = length(arr)

	if pos == -1
		pos = linspace(1, N, N)
	else
		@assert(N == size(pos,1), "Length of data != length of positions")
	end

	arr, pos, N = clean_data(arr, pos, N)

	nbins = Integer(nbins)

	if nbins > 0

		if log == true
	
			good = pos .> 0
			
			pos = pos[good] # constrain positions

			println("$(N-size(good)[1]) of $N elements not in log space")
			
			log_bnd = log10([Float64(minimum(pos)), Float64(maximum(pos))])

			bins = logspace(log_bnd[1], log_bnd[2], nbins+1)
		else 
		
			bnd = [Float64(minimum(pos)), Float64(maximum(pos))]

			bins = linspace(bnd[1], bnd[2], nbins+1)

		end

	elseif typeof(bin_low) == Array
	
		nbins = length(bin_low)-1

		bins = bin_low[1:nbins] + 0.5 .* (bin_low[2:nbins+1] - bin_los[1:nbins]) 

		bins = [minimum(pos); bins; maximum(pos)]

	else	
		@assert(false, "Cant bin without nbins > 0 or bin_low::Array")
	end
	
	bin_hig = bins[2:nbins+1]
	bin_low = bins[1:nbins]

	val = Array(Float64, nbins)
	cnt = Array(Int64, nbins)

	Threads.@threads for i = 1:nbins
	
		good = find((pos .>= bin_low[i]) & (pos .< bin_hig[i]))

		cnt[i] = size(good,1)

		if cnt[i] > 0
			val[i] = mean(arr[good])
		end

	end

	bin_pos = bin_low + 0.5 .* (bin_hig - bin_low)

	return val, bin_pos, cnt
end

function clean_data(arr::Array, pos::Array, N::Integer)

	good = find(isfinite(arr))
	
	M = length(good)

	if M < N

		println("Removing '$(N-M)' non-finite elements from the array")

		arr = arr[good]
		pos = pos[good]

		N = M
	end

	return arr, pos, N
end	

end # module
