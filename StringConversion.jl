module StringConversion

export itoa


"""
Convert integer to String of length len with zero padding

	str = itoa(4, 3)
	"003"

"""
function itoa(input::Integer, len::Integer)
	
	newstr = string(input)

	Ni = floor(log10(input))+1

	for i=1:len-Ni
		newstr = "0"*newstr
	end

	return newstr
end


end # module
