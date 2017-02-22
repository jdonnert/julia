"""
Provide some functions to compute statistics on an input array that work on
missing data.
"""
module ArrayStatistics

export asum, aminmax, amin, amax, amean, amedian, astd_dev

function asum(data::Array)
	
	return sum(data[isfinite(data)])
end

"""
Return an array containing the minimum and maximum of input array.
Works on arrays with missing data.
"""
function aminmax(data::Array)

	good = isfinite(data)

	return minimum(data[good]), maximum(data[good])
end

"""
Return  the minimum of the input array. Works on arrays with missing data.
"""
function amin(data::Array)

	return minimum(data[isfinite(data)])
end

"""
Return  the maximum of the input array. Works on arrays with missing data.
"""
function amax(data::Array)

	return maximum(data[isfinite(data)])
end

"""
Return  the mean of the input array. Works on arrays with missing data.
"""
function amean(data::Array)

	return mean(data[isfinite(data)])
end

"""
Return  the median of the input array. Works on arrays with missing data.
"""
function amedian(data::Array)

	return median(data[isfinite(data)])
end

"""
Return the standard deviation of the input array. Works on arrays with missing data.
"""
function astd_dev(data::Array)

	good = isfinite(data)

	avg = mean(data[good])
	N = length(good)

	return sqrt( sum( (data[good] .- avg).^2 ) ./ N)
end

end
