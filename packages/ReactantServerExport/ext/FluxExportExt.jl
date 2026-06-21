module FluxExportExt

using ReactantServerExport
using Reactant
using Flux

"""
    export_bundle(:flux, model, example_input; dir, name, input_name="input",
                  output_name="output", batch_sizes=[1], provenance=Dict()) -> dir

Export a Flux model. Traces `model(x)` at each batch size and writes a bundle.
"""
function ReactantServerExport.export_bundle(::Val{:flux}, model, example_input::AbstractArray;
                       dir::AbstractString, name::AbstractString,
                       input_name::AbstractString="input", output_name::AbstractString="output",
                       batch_sizes::AbstractVector{<:Integer}=[1], provenance=Dict{String,Any}())
    
    ps, re = Flux.destructure(model)
    # Convert parameters vector to an array of named weights
    wnames = ["param_$i" for i in 1:length(ps)]
    warrays = Any[ps[i] for i in 1:length(ps)]
    
    batch_axis = ndims(example_input)
    in_T = eltype(example_input)
    y0 = model(ReactantServerExport._with_batch(example_input, batch_axis, first(batch_sizes)))

    g = (x, ws...) -> re(vcat(ws...))(x)

    ctxs = Any[]
    modules = Dict{Int,Any}()
    in_shape_julia = Int[]
    for s in batch_sizes
        x = ReactantServerExport._with_batch(example_input, batch_axis, s)
        ctx = Reactant.ReactantContext()
        push!(ctxs, ctx)
        args = (Reactant.to_rarray(x), map(Reactant.to_rarray, warrays)...)
        mod, _ = Reactant.Compiler.compile_mlir(ctx, g, args; drop_unsupported_attributes=true)
        modules[Int(s)] = mod
        in_shape_julia = collect(Int, size(x))
    end

    in_batch_axis = ndims(example_input) - 1
    out_batch_axis = ndims(y0) - 1
    inputs = [ReactantServerExport.IOSpec(input_name, in_T, in_shape_julia; batch_axis=in_batch_axis)]
    outputs = [ReactantServerExport.IOSpec(output_name, eltype(y0), collect(Int, size(y0)); batch_axis=out_batch_axis)]
    prov = merge(Dict{String,Any}("source_framework" => "flux", "converter" => "ReactantServerExport.jl"),
                 Dict{String,Any}(provenance))

    GC.@preserve ctxs begin
        ReactantServerExport.write_bundle(dir; name=name, executable_inputs=inputs, executable_outputs=outputs,
            modules=modules, weights=[wnames[i] => warrays[i] for i in eachindex(wnames)],
            provenance=prov)
    end
    return dir
end

end # module
