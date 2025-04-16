# REFERENCE: https://quantum-kite.com/documentation/calculation/

# TO DO: structs GAUSSIAN WAVE PACKET


"""
    `::Configuration`
struct that contains as fields the settings passed to the KPM calculation. These are:
    - `divisions::NTuple{N, Int} = (1,1,1)` an integer number that defines
        the number of decomposition parts in each spatial direction.
         KITEx implements a domain decomposition technique  to divide the lattice into various
         partitions `divisions = (nx, ny, nz)` that are computed in parallel. Thus, multithreading
         is active when nx * ny * nz > 1. Dimensions of divisions must be those of length.
    - `unitcells::NTuple{N, Int} = (1,1,1)` number of unitcells along the x, y and z direction 
        (same dimension as divisions)
    - `boundaries::NTuple{N, String}`  Type of boundaries of the supercell
            - 'periodic' for periodic BCs
            - 'open' for open BCs
            - 'twisted' for twisted BCs
                For twisted BCs, the twist phase angles need to be specified by the user. 
                This is done by means of an extra argument: `angles=[phi_1,..,phi_DIM]` where 
                `phi_i ∈ [0, 2*M_PI]`
            - 'random' for random BCs.
                here statistical averages over ensembles of random vectors (or disorder configurations)
                are done with the help of random twist angles.  
                This special option is particularly useful to simulate the infinite-size “bulk", 
                since it supresses finite size effects. Many random vectors should be passed.
                See KITE documentation.
    - `is_complex::Bool = true` Type of the Hamiltonian entries
    - `precision::Int = 1`.
    If h matrix is real: 
        set `precision = 0` for Float32, 1 for Float64, and 2 for float128
    If h matrix is complex: 
        set `precision = 0` for ComplexF16, 1 for ComplexF32, and 2 for ComplexF64                               
    - `spectrum_range = ::Union{Nothing, Tuple{Float64, Float64}}`
        if `::Nothing` it is automatically computed
    (see: https://royalsocietypublishing.org/doi/full/10.1098/rsos.191809)

"""
struct Configuration{N, M}
    divisions::NTuple{N, Int}
    unitcells::NTuple{N, Int}
    precision::Int
    bounds_num::NTuple{N, Int}
    bounds_twist::NTuple{N, M}  # Only relevant for `boundaries = twisted` default [0.,0.] activated by boundaries != open
    spectrum_range::Union{Nothing, Tuple{M, M}}    
end

function configuration(;                               
        divisions = (1, 1), 
        unitcells = (2, 2), 
        boundaries = ("periodic", "periodic"), 
        precision = 2,
        spectrum_range = nothing,
        angles = (0., 0.))
    _division_check(unitcells, divisions)
    bounds_num, bounds_twist = _passing_bounds(boundaries, angles)
    return Configuration(divisions, unitcells, precision, bounds_num, bounds_twist, spectrum_range)
end

function _division_check(unitcells, divisions) 
    if all(unitcells .% divisions .== 0)  == false
        throw(ArgumentError("unitcells[i] / divisions[i] must be an integer"))
    else nothing end
end

""""
    `scale_shift(spectrum_range)`
energy scale and bounds for the KPM spectral boundary transformation -> [-1,1]
"""
function _scale_shift(spectrum_range)
    if spectrum_range !== nothing
        energy_scale = (spectrum_range[2] - spectrum_range[1]) / 2
        energy_shift = (spectrum_range[2] + spectrum_range[1]) / 2
    else
        energy_scale = nothing
        energy_shift = nothing
    end
    return energy_scale, energy_shift
end


 
function _passing_bounds(boundaries, angles)
    bounds_num = zeros(Int, length(boundaries))
    bound_twists = zeros(Float64, length(boundaries))
    for i in 1:length(boundaries)
        if boundaries[i] == "open"
            bounds_num[i] = 0
        elseif boundaries[i] == "periodic"
            bounds_num[i] = 1
        elseif boundaries[i] == "twisted"
            bounds_num[i] = 1
            bound_twists[i] = angles[i]
        elseif boundaries[i] == "random"
            bounds_num[i] = 2
        else
            throw(ArgumentError("Badly Defined Boundaries!"))
        end
    end
    return Tuple(bounds_num), Tuple(bound_twists)
end


####################################################################################################################################
"""
    `KPM_precision`
`num_points`: Number of energy point inside the spectrum at which the DOS will be calculated.
`num_moments`: Number of polynomials in the Chebyshev expansion.
`num_random`: Number of random vectors to use for the stochastic evaluation of trace.
`num_disorder`: Number of different disorder realisations.
"""
struct KPM_precision
    num_points::Int
    num_moments::Int
    num_random::Int
    num_disorder::Int
end

struct Dos
    settings::KPM_precision
end

dos(settings::KPM_precision) = Dos(settings)
dos(config::Configuration; num_points, num_moments, num_random, num_disorder) =
    Dos(KPM_precision(num_points, num_moments, num_random, num_disorder))

"""  
    `Ldos`
`settings::KPM_precision`
`energy`: List of energy points at which the LDOS will be calculated.
`position`: Relative index of the unit cell where the LDOS will be calculated.
`sublattice`: Name of the sublattice at which the LDOS will be calculated.
"""
struct Ldos
    settings::KPM_precision
    energy::Vector
    position::Array
    sublattice::Union{String, Array} # under development ! think about the best implementation
end
ldos(config::Configuration, num_points, num_moments, num_disorder, energy, position, sublattice) =
    Ldos(KPM_precision(num_points, num_moments, 1, num_disorder), energy, position, sublattice)
ldos(settings::KPM_precision, energy, position, sublattice) = Ldos(settings, energy, position, sublattice)

"""  
direction : string
direction in xyz coordinates along which the conductivity is calculated.
Supports 'xx', 'yy', 'zz', 'xy', 'xz', 'yx', 'yz', 'zx', 'zy'
"""
struct Conductivity_optical
    settings::KPM_precision
    direction::Int
    temperature::Float64
end

conductivity_optical(config::Configuration; num_points, num_moments, num_random, num_disorder, direction, temperature = 0.0) = 
    conductivity_optical(KPM_precision(num_points, num_moments, num_random, num_disorder), direction, temperature)

function conductivity_optical(settings::KPM_precision, direction, temperature) 
    avail_dirs = Dict("xx" => 0, "yy" => 1, "zz" => 2, "xy" => 3, "xz" => 4, "yx" => 5, "yz" => 6, "zx" => 7, "zy" => 8)
    dir = -1
    if haskey(avail_dirs, direction)
        dir = avail_dirs[direction]
    else
        throw(ArgumentError("$(direction) is not in the list of available methods: $(keys(avail_dirs))"))
    end
    return Conductivity_optical(settings, dir, temperature)
end


"""
Important: out of plane components in 2D systems are not valid..,
"""
struct Conductivity_optical_non_linear
    settings::KPM_precision
    direction::String
    temperature::Float64
    special::Int # Optional parameters, forward special, a parameter that can simplify the calculation for some materials.? Default to 0
end

conductivity_optical_non_linear(config::Configuration; num_points, num_moments, num_random, num_disorder, direction, temperature = 0.0, special = 0) = 
conductivity_optical_non_linear(KPM_precision(num_points, num_moments, num_random, num_disorder), direction, temperature, special)

function conductivity_optical_nonlinear(settings::KPM_precision, direction, temperature, special) 
    avail_dirs = Dict("xxx" => 0, "xxy" => 1, "xxz" => 2, "xyx" => 3, "xyy" => 4, "xyz" => 5, "xzx" => 6, "xzy" => 7,
    "xzz" => 8, "yxx" => 9, "yxy" => 10, "yxz" => 11, "yyx" => 12, "yyy" => 13, "yyz" => 14, "yzx" => 15,
    "yzy" => 16, "yzz" => 17, "zxx" => 18, "zxy" => 19, "zxz" => 20, "zyx" => 21, "zyy" => 22, "zyz" => 23,
    "zzx" => 24, "zzy" => 25, "zzz" => 26)
    if direction ∈ keys(avail_dirs) == false
        throw(ArgumentError("$(direction) is not in the list of available methods: $(keys(avail_dirs))"))
    else nothing end
    return Conductivity_optical_nonlinear(settings, direction, temperature, special)
end

"""
k_vector : List
List of K points with respect to reciprocal vectors b0 and b1 at which the band structure will be calculated.
weight : List
List of orbital weights used for ARPES.
"""
struct Arpes
    settings::KPM_precision
    k_vector::Array
    weight::Array
end

arpes(config::Configuration, num_moments, num_disorder, k_vector, weight) = 
    arpes(KPM_precision(1, num_moments, 1, num_disorder), k_vector, weight)

arpes(settings::KPM_precision, k_vector, weight) = Arpes(settings,  k_vector, weight)

struct Conductivity_dc
    settings::KPM_precision
    direction::Int #JAP changed to Int to be coherent with KITE
    temperature::Float64
end


conductivity_dc(config::Configuration; num_points, num_moments, num_random, num_disorder, direction, temperature = 0.0) = 
    conductivity_dc(KPM_precision(num_points, num_moments, num_random, num_disorder), direction, temperature)

function conductivity_dc(settings::KPM_precision, direction, temperature)  #JAP changed code to be equal to conductivity_optical
    avail_dirs = Dict("xx" => 0, "yy" => 1, "zz" => 2, "xy" => 3, "xz" => 4, "yx" => 5, "yz" => 6, "zx" => 7, "zy" => 8)
    dir = -1
    if haskey(avail_dirs, direction)
        dir = avail_dirs[direction]
    else
        throw(ArgumentError("$(direction) is not in the list of available methods: $(keys(avail_dirs))"))
    end
    return Conductivity_dc(settings, dir, temperature)
end

"""
`eta`: Parameter that affects the broadening of the kernel function
`preserve_disorder = false`
"""
struct Singleshot_conductivity_dc
    settings::KPM_precision
    energy::Union{Array, Float64}
    direction::String
    eta::Float64
    temperature::Float64
    preserve_disorder::Bool

end


singleshot_conductivity_dc(config::Configuration, num_moments, num_random, num_disorder, energy, direction, eta, preserve_disorder = false) = 
    singleshot_conductivity_dc(KPM_precision(1, num_moments, num_random, num_disorder),  energy, direction, eta, preserve_disorder)

function singleshot_conductivity_dc(settings::KPM_precision, energy, direction, eta, preserve_disorder) 
    avail_dirs = Dict("xx" => 0, "yy" => 1, "zz" => 2)
    if direction ∈ keys(avail_dirs) == false
        throw(ArgumentError("$(direction) is not in the list of available methods: $(keys(avail_dirs))"))
    else nothing end
    return Singleshot_conductivity_dc(settings, energy, direction, eta, preserve_disorder)
end

"""
        Calculate the time evolution function of a wave packet
`timestep`: Timestep for calculation of time evolution.
`k_vector`: Different wave vectors, components corresponding to vectors b0 and b1.
`spinor`: Spinors for each of the k vectors.
`width`: Width of the gaussian.
`mean_value`: Mean value of the gaussian envelope.
  Optional parameters, forward `probing point`, defined with x, y coordinate were the wavepacket will be checked
    at different timesteps.
"""
struct Gaussian_wave_packet
    settings::KPM_precision
    timestep::Float64
    k_vector::Array
    spinor::Array
    width::Float64
    mean_value::Array
    probing_point::Int
end

gaussian_wave_packet(config::Configuration, num_points, num_moments, num_disorder, timestep, k_vector, spinor, width, mean_value, probing_point = 0) = 
    gaussian_wave_packet(KPM_precision(num_points, num_moments, 1, num_disorder), timestep, k_vector, spinor, width, mean_value, probing_point)

gaussian_wave_packet(settings::KPM_precision, timestep, k_vector, spinor, width, mean_value, probing_point ) =
    Gaussian_wave_packet(settings, timestep, k_vector, spinor, width, mean_value, probing_point )

# HDF5 summary
# fid = h5open(fname, "w")
# create_group(fid, "mygroup")
# fid["mydataset"] = rand()
#  create_dataset(fid, "myvector", Int, (10,))
# g = fid["mygroup"]
# g["mydataset"] = "Hello World!"
# g = create_group(parent, name) The named group will be created as a child of the parent.
# attributes(parent)[name] = value
# read_attribute(parent, name)
#  attr = attributes(parent)[name];
