module Binning

export BinArr

function BinArr(arr::Array; pos=-1, bin_low=-1, log=false, nbins=0)

	@assert(ndims(arr) == 1, "\nCan't bin multi-dimensional arrays\n")

	N = (size(arr))[1]

	if pos == -1
		pos = linspace(1, N, N)
	end

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

	elseif bin_low > 0
	
		nbins = length(bin_low)-1

	else	
		@assert(false, "Cant bin without nbins or bin_pos")
	end
	
	bin_hig = bins[2:nbins+1]
	bin_low = bins[1:nbins]

	bin_pos = bin_low .+ 0.5 .* (bin_hig .- bin_low)

	val = SharedArray(Float64, nbins)
	cnt = SharedArray(Int64, nbins)

	Threads.@threads for i = 1:nbins
	
		good = find((pos .>= bin_low[i]) &  (pos .<= bin_hig[i]))

		cnt[i] = length(good)

		if cnt[i] > 0
			
			val[i] = mean(arr[good])
		end

	end

	return Array(val), bin_pos, Array(cnt)
end

end # module
