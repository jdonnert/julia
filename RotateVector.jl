module RotateVector

export rotation_matrix_YPR3D, rotation_matrix_EulerAng


"""
Rotation matrix of Luftfahrtnorm (DIN 9300) (Yaw-Pitch-Roll, Z, Y’, X’’)

    function rotation_matrix_YPR3D(psi::Real, theta::Real, phi::Real; elType=Float64)

psi first rotates around z-axis
theta then rotates around the new y-axis
phi then rotates around new x-axis

Angles in radiants

Rotate a vector with 

    M = rotation_matrix_YPR3D(pi/2,0,0)
    b = M * a

See Wikipedia
"""
function rotation_matrix_YPR3D(psi::Real, theta::Real, phi::Real; elType=Float64)

    M = Array{elType}(3,3)
    
    M[1,1] = cos(theta) * cos(psi)
    M[2,1] = cos(theta) * sin(psi)
    M[3,1] = -sin(theta)
         
    M[1,2] = sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi)
    M[2,2] = sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi)
    M[3,2] = sin(phi) * cos(theta)
         
    M[1,3] = cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)
    M[2,3] = cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi)
    M[3,3] = cos(phi) * cos(theta)

    return M
end

"""
Rotation matrix for Euler angles, y-Convention

    function rotation_matrix_EulerAng3D(alpha::Real, beta::Real, gamma::Real)

alpha first rotates around z-axes
beta the rotates around the new y-axis
gamma then rotates around the new z-axes

Angles in radiants

Rotate a vector with 

    M = rotation_matrix_EulerAng3D(pi/2,0,0)
    b = M * a

See Wikipedia
"""
function rotation_matrix_EulerAng3D(alpha::Real, beta::Real, gamma::Real)

    M =  Array{elType}(3,3)
    
    M[1,1] = -sin(alpha) * sin(gamma) + cos(alpha) * cos(beta) * cos(gamma) 
    M[2,1] = -sin(alpha) * cos(gamma) - cos(alpha) * cos(beta) * sin(gamma) 
	M[3,1] = cos(alpha) * sin(beta)

	M[1,2] = cos(alpha) * sin(gamma) + sin(alpha) * cos(beta) * cos(gamma)
	M[2,2] = cos(alpha) * cos(gamma) - sin(alpha) * cos(beta) * sin(gamma)
	M[3,2] = sin(alpha) * sin(beta)

	M[1,3] = -sin(beta) * cos(gamma)
    M[2,3] = sin(beta) * sin(gamma)
    M[3,3] = cos(beta)
    
    return Array{Float64}(M)
end


end
