module QuantiKite
    # Use README as the docstring of the module:
    @doc read(joinpath(dirname(@__DIR__), "README.md"), String) QuantiKite
    using HDF5, Arpack, Quantica, SparseArrays, DelimitedFiles, CairoMakie
    export h5gen, configuration, dos, conductivity_optical, conductivity_optical_non_linear,
    dos_plot, linear_optical_conductivity_plot, conductivity_dc
    export Configuration, Dos, Ldos, Conductivity_optical, Conductivity_dc, 
        Conductivity_optical_non_linear, Singleshot_conductivity_dc, Arpes
    include("structures.jl")
    include("kite_wrapper.jl")
    include("plotting_functions.jl")
end
