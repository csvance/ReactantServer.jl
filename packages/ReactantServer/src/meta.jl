# Meta-model execution: a meta model is a user-authored Julia workflow (a bundle's model.jl that
# calls register_meta_model) that chains several models with data-dependent logic in between. A meta
# is a scheduled unit: the dispatch loop selects it and runs its orchestration INLINE, holding the
# GPU exclusively for the whole run (see scheduler.jl `execute_meta!`). Its sub-calls go through an
# injected `InlineCaller` that invokes each sub-model's compiled executable directly in-process, with
# no queue re-entry, no loopback gRPC, and no shared-memory transport. The sub-models are internal to
# the meta (the gateway never routes to them); placement keeps them resident with their meta.

abstract type ModelCaller end

# In-process caller. A sub-call runs the sub-model's compiled executable directly on the dispatch
# thread: this is exactly `infer` minus the queue (preprocess, ensure weights resident, run_model,
# postprocess). Because the meta holds the GPU exclusively while it runs, these calls execute back to
# back with no coalescing and no transport. Carries everything `run_model`/`acquire!` need, plus the
# worker's local reuse pool for `call.scratch`.
struct InlineCaller <: ModelCaller
    backend::AbstractBackend
    pool::MemoryPool
    registry::ModelRegistry
    weight_cache::Union{WeightCache,Nothing}
    scratch::Union{BufferPool,Nothing}
end

# The local reuse pool a meta's `call.scratch` allocates from (or nothing -> plain heap arrays).
caller_pool(c::InlineCaller) = c.scratch

# `deadline_ns` is an absolute local `time_ns()` deadline (0 = none). The sub-model runs in-process,
# so the deadline is just checked here before starting more GPU work; there is no hop to propagate it
# across. `requested_outputs` is accepted for interface symmetry but unused: `run_model` returns all
# outputs and the meta selects what it needs by position/name.
function call_model(c::InlineCaller, name::AbstractString, inputs::Vector{NamedTensor};
                    requested_outputs::Vector{String}=String[], deadline_ns::Integer=0)
    sub = get(c.registry.by_name, String(name), nothing)
    (sub === nothing || sub.executable === nothing) && error("unknown model: $name")
    deadline_ns != 0 && Int64(time_ns()) >= deadline_ns && throw(DeadlineExceeded(String(name)))
    # preprocess/postprocess come from the sub-model's model.jl (a newer world age); invokelatest
    # crosses that boundary, exactly as infer() does. acquire! ensures the sub-model's weights are
    # resident on this worker (a fast no-op when placement co-resides the meta's group).
    prepared = Base.invokelatest(sub.preprocess, inputs)
    c.weight_cache === nothing || acquire!(c.weight_cache, sub)
    raw = run_model(c.backend, c.pool, sub.executable, prepared)
    return Base.invokelatest(sub.postprocess, raw)
end

# The injected `call` handed to a meta model's `run(inputs, call)`. Callable to dispatch a sub-call
# (`call(name, inputs)`), and exposes `call.scratch(...)` to request reuse buffers. Tracks the pool
# slots acquired this request so `run_meta` releases them when the orchestration returns.
mutable struct MetaCall{C<:ModelCaller}
    caller::C
    name::String
    declared::Set{String}
    slots::Vector{PoolSlot}
    scratched::Bool          # call.scratch may be used at most once per request (see _scratch)
    deadline_ns::Int64       # absolute local time_ns() deadline for the whole orchestration (0 = none)
end
MetaCall(caller::ModelCaller, name::AbstractString, declared; deadline_ns::Integer=0) =
    MetaCall(caller, String(name), Set{String}(declared), PoolSlot[], false, Int64(deadline_ns))

# `call(name, inputs)` — dispatch a sub-call, rejecting undeclared callees. Bail before issuing if
# the orchestration's deadline has passed (no point starting more GPU work the caller has abandoned),
# and pass the deadline down so the in-process sub-call short-circuits too.
function (mc::MetaCall)(name::AbstractString, inputs::Vector{NamedTensor};
                        requested_outputs::Vector{String}=String[])
    String(name) in getfield(mc, :declared) ||
        error("meta model '$(getfield(mc, :name))' called undeclared model '$name'; add it to meta.calls")
    dl = getfield(mc, :deadline_ns)
    dl != 0 && Int64(time_ns()) >= dl && throw(DeadlineExceeded(getfield(mc, :name)))
    return call_model(getfield(mc, :caller), name, inputs;
                      requested_outputs=requested_outputs, deadline_ns=dl)
end

# `call.scratch(...)` resolves to a closure over this MetaCall; every other field reads normally.
function Base.getproperty(mc::MetaCall, s::Symbol)
    s === :scratch && return (args...) -> _scratch(mc, args...)
    return getfield(mc, s)
end

"""
    call.scratch(dims, T) -> Array{T}
    call.scratch([dims1 => T1, dims2 => T2, ...]) -> Vector{Array}

Request reuse buffers for this request. With a pool, all requested buffers are carved from ONE
contiguous acquired block (released when the orchestration returns); without a pool, plain heap
arrays are returned. Identical model.jl code either way. The pool exists purely to keep large
intermediates (e.g. an ROI feature tensor) off the per-request allocation path and hold down GC
pressure: a meta writes into the pooled buffer and hands it to the next stage by reference (the call
is in-process, so there is no copy). The pool is plain `Memory`, local to the worker, never shared.

Call this at most once per request — ask for ALL buffers up front in a single call (pass a vector of
`dims => T`). One call carves N buffers from ONE block (O(1) waste); N separate calls would grab N
blocks. A second call throws.
"""
_scratch(mc::MetaCall, dims, ::Type{T}) where {T} =
    first(_scratch(mc, Pair[(dims isa Tuple ? dims : (dims,)) => T]))

function _scratch(mc::MetaCall, reqs::AbstractVector)
    getfield(mc, :scratched) && error(
        "meta model '$(getfield(mc, :name))': call.scratch may be called only once per request; " *
        "request all buffers up front in a single call, e.g. call.scratch([dims1 => T1, dims2 => T2]).")
    setfield!(mc, :scratched, true)
    specs = [(Tuple(first(r)), last(r)) for r in reqs]   # dims => T pairs
    pool = caller_pool(getfield(mc, :caller))
    if pool === nothing
        return Any[Array{T}(undef, d...) for (d, T) in specs]
    end
    # One contiguous block fits everything (simple by design; revisit for fragmentation later). Since
    # metas run exclusively, the pool sees no cross-meta contention; the deadline-bounded acquire is a
    # harmless guard rather than a load-bearing back-pressure mechanism.
    total = sum(sizeof(T) * prod(d) for (d, T) in specs)
    span = max(1, cld(total, pool_slot_bytes(pool)))
    slot = try
        acquire_slot!(pool, span; deadline_ns = getfield(mc, :deadline_ns))
    catch e
        e isa PoolAcquireTimeout && throw(DeadlineExceeded(getfield(mc, :name)))
        rethrow()
    end
    push!(getfield(mc, :slots), slot)
    return Any[pool_view(subslot(slot, sizeof(T) * prod(d)), T, d...) for (d, T) in specs]
end

"""
    run_meta(entry, caller, inputs; deadline_ns=0) -> Vector{NamedTensor}

Run a meta model's orchestration. The injected `call` (a [`MetaCall`](@ref)) dispatches sub-calls
through `caller` (an [`InlineCaller`](@ref) on the dispatch thread), rejecting any callee not
declared in `meta.calls`, and offers `call.scratch` for reuse buffers. `deadline_ns` is an absolute
local `time_ns()` deadline for the whole orchestration (0 = none): once it passes, the next
`call(...)` (or a slot acquire) bails with [`DeadlineExceeded`](@ref) rather than starting more GPU
work. Any pool slots acquired via `scratch` are released here when it returns (request-scoped).
"""
function run_meta(entry::MetaEntry, caller::ModelCaller, inputs::Vector{NamedTensor};
                  deadline_ns::Integer=0)
    mc = MetaCall(caller, entry.name, entry.calls; deadline_ns=deadline_ns)
    try
        # The orchestration is defined in a sandboxed model.jl (a newer world age), so cross it with
        # invokelatest, exactly as infer() does for pre/post hooks.
        out = Base.invokelatest(entry.run, inputs, mc)
        out isa Vector{NamedTensor} ||
            error("meta model '$(entry.name)' returned $(typeof(out)); expected Vector{NamedTensor}")
        return out
    finally
        for s in getfield(mc, :slots)
            release_slot!(s)
        end
    end
end
