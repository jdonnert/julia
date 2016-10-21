module Binning

export BinArr

function BinArr(arr::AbstractArray; pos=0, bin_low=0, log=false, nbins=0)

	@assert(ndims(arr) == 1, "\nCan't bin multi-dimensional arrays\n")

	N = (size(arr))[1]

	if pos == 0
		pos = linspace(1, N, N)
	end

	if nbins > 0

		if log == true
	
			good = pos .> 0
			
			pos = pos[good] # constrain positions

			println("$(N-size(good)[1]) of $N elements not in log space")
			
			log_bnd = log10([minimum(pos), maximum(pos)])

			bin_low = logspace(log_bnd[1], log_bnd[2], nbins+1)
		else 
		
			bnd = [minimum(pos), maximum(pos)]

			bin_low = linspace(bnd[1], bnd[2], nbins+1)

		end

	elseif bin_pos != 0
	
		nbins = size(bin_low)-1

	else	
		@assert(false, "Cant bin without nbins or bin_pos")
	end
	
	i = 1:nbins

	bin_hig = bin_low[i+1]

	bin_pos = bin_low[i] + 0.5 * (bin_hig[i] - bin_low[i])

	val = zeros(Float64, nbins)
	cnt = zeros(Float64, nbins)

	for i = 1:nbins
	
		good = find((pos .>= bin_low[i]) &  (pos .<= bin_hig[i]))

		cnt[i] = length(good)

		if cnt[i] > 0
			
			val[i] = mean(arr[good])
		end

	end

	return val, bin_pos, cnt
end

end # module
