
#-------------------------------------------------------------------------------------------------
#                     THIS CODE IS INSPIRED BY THE KITE.PY FILE IN KITE.
#                  REFERENCE: https://quantum-kite.com/documentation/calculation/
#-------------------------------------------------------------------------------------------------



const Methods = Union{Dos, Ldos, Conductivity_optical, Conductivity_dc, 
    Conductivity_optical_non_linear, Singleshot_conductivity_dc, Arpes}

scale_and_shift(h, unitcells, (emin, emax), padfactor = 0.9) = (emax - emin)/2, (emin + emax)/2

"""
 If `scale_energy`` and `shift_energy` are missing in the config struct they are automatically
 computed using Quantica machinery.
"""
grow_and_bound!(h, sx, sy, ::Missing) =  h |> supercell(sx, sy) |> supercell()
grow_and_bound!(h, sx, sy, sz::Int64) =  h |> supercell(sx, sy, sz) |> supercell()

function scale_and_shift(h::Quantica.Hamiltonian, unitcells::NTuple, ::Nothing, padfactor = 0.9)
    @warn "Computing spectrum bounds... Consider passing a bandrange for faster performance."
    sz::Union{Int, Missing} = missing
    if length(unitcells) == 2
        sx, sy, sz =  unitcells[1], unitcells[2], missing
    elseif length(unitcells) == 3
        sx, sy, sz =  unitcells[1], unitcells[2], unitcells[2]
    else
        throw(ArgumentError("1D not implemented..."))
    end
    ϵmin, ϵmax = custom_bandrange_arpack(grow_and_bound!(h, sx, sy, sz)[()])
    return (ϵmax - ϵmin) / (2 * padfactor), (ϵmax + ϵmin) / 2
end

function custom_bandrange_arpack(h::AbstractMatrix{T}) where {T} 
    R = real(T)
    ϵL, _ = Arpack.eigs(h, nev=1, tol=1e-4, which = :LR)
    ϵR, _ = Arpack.eigs(h, nev=1, tol=1e-4, which = :SR)
    ϵmax = R(real(ϵL[1]))
    ϵmin = R(real(ϵR[1]))
    return (ϵmin, ϵmax)::Tuple{R, R}
end

"""
    `h5gen(h::Quantica.Hamiltonian, c::Configuration, s::T, modification = false; kws...)`
returns an h5 file containing all the information required by KITE to compute the observable
encoded by `s::Methods`, i.e.: Dos, Ldos, Conductivity_optical, Conductivity_dc, 
Conductivity_optical_non_linear, Singleshot_conductivity_dc, or Arpes`. `h::Quantica.Hamiltonian`
contains information about the unitcell and `c::Configuration` further instructions about the 
calculation.

See: `Configuration`, `Methods`
"""
function h5gen(h::Quantica.Hamiltonian, c::Configuration, s::T, modification = false;
        kws...)  where {T <: Methods} #JAP if s is an instance of a Method, call h5gen with it in an array 

    h5gen(h, c, [s], modification;kws...)
end

#JAP ss is now an array of Methods to support different methods in same hdf5 file
function h5gen(h::Quantica.Hamiltonian, c::Configuration, ss::T, modification = false;
        kws...)  where {T <: AbstractArray} #JAP ss is an array of methods

    # reads the type of h or promote it to complex if needed #0 real, #1 complex
    complx = real_or_complex(h, ss) 
    # sets the precision of the calculation using complx type and the `precision` field in c
    precision = set_prec(complx, c.precision) 
    # energy rescaling for KPM computations.
    energy_scale, energy_shift = scale_and_shift(h, c.unitcells, c.spectrum_range) 
    # this bandwidth is an approximation to the larger system
    vectors = h.lattice.bravais.matrix' # bravais vectors
    space_size = size(vectors,1)        # system dimension
    #print(h.lattice.unitcell.sites)
    position = site_positions(h)        # orbital positions (degenerate)
    orb_from, orb_to, values = hdf5_rearrangefunction(h, energy_scale, energy_shift, space_size)
    ts, ds, num_hoppings_orbital = matrix_elements_and_distances(orb_from, orb_to, values)       

    #---------------------------------------------------------------------------- HAMILTONIAN GROUP
    
    filename = return_kwargs(kws, :filename)
    filename = ifelse(isa(filename, Nothing), "kite_config.h5", filename)
    disorder = return_kwargs(kws, :disorder)
    disorder_structural = return_kwargs(kws, :disorder_structural)


    f = h5open(filename, "w")
    f["IS_COMPLEX"] = UInt32(complx)
    f["PRECISION"] =  UInt32(c.precision)
    f["Divisions"] = [UInt32(x) for x in c.divisions]
    f["DIM"] = UInt32(space_size) # space dimension of the lattice 1D, 2D, 3D
    f["LattVectors"] = vectors.parent
    #print(Matrix(hcat(position...)))
    f["OrbPositions"] = Matrix(hcat(position...)) 
    #f["NOrbitals"] = UInt32(sum(Quantica.norbitals(h)))
    f["NOrbitals"] = UInt32(get_num_orbitals(h)) #JAP this gets the total num of orbitals and not just sublattices 
    f["EnergyScale"] = Float64(energy_scale)
    f["EnergyShift"] = Float64(energy_shift)
    
    f["L"] = convert(Array, [UInt32(c.unitcells[i]) for i in 1:length(c.unitcells)])
    if length(c.unitcells) != space_size
        throw(ArgumentError("dim(unitcells) = sys_dims"))
    else nothing end
    # periodic boundary conditions, 0 - no, 1 - yes, 2 -> randomly twisted
    f["Boundaries"] = [UInt32(i) for i in c.bounds_num]
    f["BoundaryTwists"] = [float(i) for i in c.bounds_twist]
    
    # Hamiltonian group
    grp = create_group(f, "Hamiltonian")
    grp["NHoppings"] =  [UInt32(x) for x in num_hoppings_orbital]
    grp["d"] = convert.(Int32, ds')
    grp["Hoppings"] = convert(Matrix{precision}, ts') 
    grp["CustomLocalEnergy"] = 0
    grp["PrintCustomLocalEnergy"] = 0
    
    # !!! The disorder group has to be specified in advance. Defaulted to none
    grp_dis = create_group(grp, "Disorder")   
    datatype = Float32
    dset = create_dataset(grp_dis, "OnsiteDisorderMeanStdv", datatype, (1,0))
    write(dset, zeros(Float64, 0, 1))
    dset = create_dataset(grp_dis, "OnsiteDisorderMeanValue", datatype, (1,0))
    write(dset, zeros(Float64, 0, 1))
    dset = create_dataset(grp_dis, "OnsiteDisorderModelType", datatype, (1,0))
    write(dset, zeros(Float64, 0, 1))
    dset = create_dataset(grp_dis, "OrbitalNum", datatype, (1,0))
    write(dset, zeros(Float64, 0, 1))

    #---------------------------------------------------------------------------- CALCULATION GROUP
    grpc = create_group(f, "Calculation")
   for s in ss  
        if isa(s, Dos)
            grpc_p = create_group(grpc, "dos")
            grpc_p["NumMoments"] = [Int32(s.settings.num_moments)]
            grpc_p["NumPoints"] = [Int32(s.settings.num_points)]
            grpc_p["NumRandoms"] = [Int32(s.settings.num_random)]
            grpc_p["NumDisorder"] = [Int32(s.settings.num_disorder)] 
        
        elseif isa(s, Conductivity_optical)
            grpc_p = create_group(grpc, "conductivity_optical")
            grpc_p["NumMoments"] = [Int32(s.settings.num_moments)]
            grpc_p["NumPoints"] = [Int32(s.settings.num_points)]
            grpc_p["NumRandoms"] = [Int32(s.settings.num_random)]
            grpc_p["NumDisorder"] = [Int32(s.settings.num_disorder)] 
            grpc_p["Temperature"] =  [s.temperature/energy_scale] # KPM rescaled
            grpc_p["Direction"] = [Int(s.direction)]

        elseif isa(s, Conductivity_optical_non_linear)
            grpc_p = create_group(grpc, "conductivity_optical_nonlinear")
            grpc_p["NumMoments"] = [Int32(s.settings.num_moments)]
            grpc_p["NumPoints"] = [Int32(s.settings.num_points)]
            grpc_p["NumRandoms"] = [Int32(s.settings.num_random)]
            grpc_p["NumDisorder"] = [Int32(s.settings.num_disorder)] 
            grpc_p["Temperature"] =  [s.temperature/energy_scale] # KPM rescaled
            grpc_p["Direction"] = [Int(s.direction)]
            grpc_p["Special"] = [Int(s.special)]
        else nothing end
    end
    close(f)
    println("A .h5 file has been generated")
end

function real_or_complex(h, ss::T) where {T<:AbstractArray} #JAP added AbstractArray for same reason as above
    hamtype = eltype(h.harmonics[1].h.flat)
    if hamtype ∈ [ComplexF16, ComplexF32, ComplexF64]  
        complx = 1
    else
        complx = 0
    end
    for s in ss
        if isa(s, Arpes) && complx == 0
            print("ARPES is requested but is_complex identifier is 0.
                Automatically turning is_complex to 1!")
            complx = 1
        elseif isa(s ,Gaussian_wave_packet) && complx == 0
            print("Wavepacket is requested but is_complex identifier is 0. 
                Automatically turning is_complex to 1!")
            complx = 1
        else nothing end
    end
    return complx # if hamiltonian is complex complx = 1 else complx = 0
end

"""
This struct stores all the hoppings between the elements connecting two unit cells with relative 
index dn = [i,j] if i = j = 0 corresponding to the matrix elements between orbitals inside the unitcell,
it also stores the onsite energies"""
struct Same_harm{N,T}
    dn                          # harmonic coordinates [dn[1], dn[2]]. Unitcell at [0,0]
    orb_from::Array{N, 1}       # orbital of origin `i` in `h[(dn[1], dn[2])][i, j]`
    orb_to::Array{N, 1}         # orbital of destination `j` in `h[(dn[1], dn[2])][i, j]`
    vals::Array{T, 1}           # hopping or onsite energy `h[(dn[1], dn[2])][i, j]`
end                             # note that onsite energies should be shifted before being rescaled

"""
this struct stores Same_harmonics structs for all dns in h::Quantica.Hamiltonian
"""
struct Hdf5_h
    s::Vector{Same_harm}
end

function site_positions(h) # site positions for each orbital in the TB matrix   
    # it accepts systems with different orbitals at different sites
    positions = []
    position_atoms = h.lattice.unitcell.sites
    num_orbitals = Quantica.norbitals(h,:) # vector of num_orbitals at site i JAP norbitals has now two methods pertaining hamiltonians in newer Quantica versions. putting the : gets the vector with the n of orbitals. without the collon it only gives the size of the vector. 
    print("\n")
    print(Quantica.norbitals(h,:))
    print("\n")
    num_sites = length(position_atoms)
    num_sublat = length(num_orbitals)
    print(num_sublat)
    print("\n")
    chunks = Iterators.partition(1:num_sites,div(num_sites,num_sublat))
    for (sublat_ind,chunk) in enumerate(chunks)
        for i in chunk
            for j in 1:num_orbitals[sublat_ind]
                push!(positions, position_atoms[i])
            end
        end
    end
    return positions
end

function get_num_orbitals(h) #JAP Calculate total number of orbitals in unit cell 

    orb_per_sublat = h.blockstruct.subsizes
    atomic_orb = h.blockstruct.blocksizes

    return orb_per_sublat' * atomic_orb
end

function hdf5_rearrangefunction(h, energy_scale, energy_shift, space_size)
    h_info = hdf5_hamiltonian(h, energy_scale, energy_shift)
    # list of origin orbital indices for all dns
    orb_from = reduce(vcat, [harm.orb_from for harm in h_info.s]) .-1 
    vals = reduce(vcat, [harm.vals for harm in h_info.s])
    orb_to = []
    for harm in h_info.s
        rel_move = dot((harm.dn .+1), 3 .^(collect(0:space_size-1)))
        orb_to_harm = (harm.orb_to .-1) .* 3^space_size .+ rel_move
        append!(orb_to, orb_to_harm)
    end
    return orb_from, orb_to, vals
end

function hdf5_hamiltonian(h, energy_scale, energy_shift) #vals are KPM rescaled
    har_dim = length(h.harmonics)
    dn_resolved_info = []
    for har in Quantica.harmonics(h) 
        dn = Quantica.dcell(har)
        orb_from, orb_to, vals = findnz(dropzeros(copy(h[dn]'))) # JAP added transpose here
        onsite_indx = findall(x-> x == 0, orb_from - orb_to)
        vals[onsite_indx] .-= energy_shift/energy_scale # rescaled shift for onsites
        push!(dn_resolved_info, Same_harm(dn, orb_from, orb_to, vals./ energy_scale))
    end
    return Hdf5_h(dn_resolved_info)
end

"""
We need to pass the information to the hdf5 in a very precise manner so we can use the Kitex
code without further modifications. This is done in this function which is suited for the general
case of multiorbital systems where different sites can host different number of orbitals. The output
are the two arrays `t` and `d` containing the Hamiltonian matrix elements (onsites included) and their
indices in an unique way (see Kite.py documentaion). It is valid for 2D and 3D systems.
"""
function matrix_elements_and_distances(orb_from, orb_to, vals)
    smat = sparse(orb_from .+ 1, orb_to .+ 1, vals) #cuidado con los indices + 1 in orb_to?? //JAP added +1 to orb_to here, remove it later in cols, so that harmonic [-1,-1] (index 0) does not give an error here

    # fix the size of hoppings and distance matrices where the number of columns is max number of 
    # hoppings this step is necessary to include the most general multiorbital case, i.e., sites 
    # with different number of orbitals
    num_hoppings_orbital = [nnz(smat[i,:]) for i in 1:size(smat,1)]
    max_orb_connectivity = maximum(num_hoppings_orbital)
    t_list = []
    d_list = []

    for i in 1:size(smat,1)
        push!(t_list, nonzeros(smat[i,:])) # make it real if smat is real.
        cols, _ = findnz(smat[i,:])
        cols .-= 1 #JAP now I remove the previous +1 to be coherent with KITE
        push!(d_list, cols)
    end
    
    t = zeros(eltype(smat),size(smat,1), max_orb_connectivity)       # add type #JAP added type of smat
    d = zeros(Int, size(smat,1), max_orb_connectivity)  # add type

    for (i_row, d_row) in enumerate(d_list)
        d[i_row, 1:length(d_row)] .= Int32.(d_row)
        t[i_row, 1:length(d_row)] .= t_list[i_row]
    end
    return t, d, num_hoppings_orbital

end

function set_prec(is_complex, precision)
    if is_complex == 0
        if precision == 0
        htype = Float32
        elseif precision == 1
        htype = Float64
        elseif precision == 2
        htype= Float128
        else
        throw(ArgumentError("Precision should be 0, 1 or 2"))
        end
    else
        if precision == 0
            htype = ComplexF16
        elseif precision == 1
            htype = ComplexF32
        elseif precision == 2
            htype = ComplexF64
        else
            throw(ArgumentError("Precision should be 0, 1 or 2"))
        end
        end
    return htype
end

function return_kwargs(kws, name::Symbol)
    kwargs = NamedTuple(kws)
    if haskey(kwargs, name)
        return kwargs[name]
    else  nothing
    end
end
