# Periodic worker readiness probe and route autodiscovery. Each round probes every endpoint's
# ServerReady (driving /readyz: the aggregate is ready when at least one worker reports ready) and
# its RepositoryIndex (driving the routing table: the model -> ready-endpoints map is rebuilt and
# swapped in atomically). A control-plane pin/unpin or a node restart is picked up on the next
# round. Mirrors and extends the Go gateway's internal/health.

const HEALTH_INTERVAL_SECONDS = 10.0

# Watchdog ceiling for one probe/discovery call. The gRPC clients carry their own deadline, but
# that deadline is enforced by the shared libcurl multi handle, and the multi's timer/socket
# driving can wedge after a burst of failed connects (e.g. the window where every worker is
# still compiling). A wedged multi hangs every call forever; an unbounded call inside the
# prober's @sync would then wedge the round, and with it /readyz, permanently. A timed-out call
# counts as not-ready / no-routes and is retried next round (the abandoned task is leaked,
# bounded to worker-churn windows).
const PROBE_TIMEOUT_SECONDS = 8.0

# A watchdog timeout is also the wedge signature: the client's own 5s deadline can only be
# missed when the multi is no longer being driven at all. A fresh process always recovers, so
# after this many consecutive all-timeout probe rounds the gateway exits and its supervisor
# (the node supervisor, or the container restart policy) restarts it. Override with
# REACTANT_GATEWAY_WEDGE_EXIT_ROUNDS; 0 disables.
const WEDGE_EXIT_ROUNDS = 3

# Returns (value, timed_out). On timeout the abandoned task keeps running detached; `default`
# is returned in its place.
function _bounded(f, secs, default, what::String, url::String)
    t = @async f()
    if timedwait(() -> istaskdone(t), secs) !== :ok
        @warn "health: $what timed out; treating as unavailable" worker = url timeout_s = secs
        return default, true
    end
    v = try
        fetch(t)
    catch
        default
    end
    return v, false
end

mutable struct HealthProber
    pool::ClientPool
    metrics::GatewayMetrics
    admin::AdminServer
    routes::Union{DiscoveredRoutes,Nothing}
    packing::Union{LptPackingState,Nothing}   # packing-mode rebalancer, ticked on its own interval
    interval::Float64
    running::Threads.Atomic{Bool}
    task::Union{Task,Nothing}
    wedged_rounds::Int                        # consecutive rounds where every probe hit the watchdog
    wedge_exit_rounds::Int                    # exit(1) after this many; 0 disables
end

function HealthProber(pool::ClientPool, metrics::GatewayMetrics, admin::AdminServer,
                      routes::Union{DiscoveredRoutes,Nothing} = nothing;
                      packing::Union{LptPackingState,Nothing} = nothing,
                      interval::Real = HEALTH_INTERVAL_SECONDS,
                      wedge_exit_rounds::Integer =
                          parse(Int, get(ENV, "REACTANT_GATEWAY_WEDGE_EXIT_ROUNDS",
                                         string(WEDGE_EXIT_ROUNDS))))
    return HealthProber(pool, metrics, admin, routes, packing, Float64(interval),
                        Threads.Atomic{Bool}(true), nothing, 0, Int(wedge_exit_rounds))
end

# Query every endpoint's ready models concurrently and build the model -> endpoints routing table.
# An unreachable endpoint is skipped (it contributes no routes this round and is picked up later).
function discover_routes(pool::ClientPool)
    workers = all_clients(pool)
    found = Dict{String,Vector{String}}()
    lk = ReentrantLock()
    @sync for wc in workers
        @async begin
            names, _ = _bounded(() -> discover_models(wc), PROBE_TIMEOUT_SECONDS, nothing,
                                "RepositoryIndex discovery", wc.url)
            names === nothing && return
            lock(lk) do
                for n in names
                    push!(get!(found, n, String[]), wc.url)
                end
            end
        end
    end
    return RoutingTable(found)
end

function _check_once(p::HealthProber)
    workers = all_clients(p.pool)
    results = Vector{Bool}(undef, length(workers))
    timeouts = Vector{Bool}(undef, length(workers))
    @sync for (i, wc) in enumerate(workers)
        @async results[i], timeouts[i] =
            _bounded(() -> probe_ready(wc), PROBE_TIMEOUT_SECONDS, false,
                     "ServerReady probe", wc.url)
    end
    any_ready = false
    for (i, wc) in enumerate(workers)
        set_worker_ready!(p.metrics, wc.url, results[i])
        any_ready |= results[i]
    end
    set_ready!(p.admin, any_ready)
    _track_wedge!(p, !isempty(workers) && all(timeouts))
    if p.routes !== nothing
        table = discover_routes(p.pool)
        swap_table!(p.routes, table)
        set_routing_size!(p.metrics, nmodels(table))
    end
    # LPT-packing rebalance on its own cadence (>= the probe interval). Placement is computed over
    # the workers that reported ready this round; a dead worker drops out until it recovers.
    if p.packing !== nothing &&
       time() - p.packing.last_rebalance >= p.packing.rebalance_seconds
        ready_urls = String[wc.url for (i, wc) in enumerate(workers) if results[i]]
        if isempty(ready_urls)
            @warn "lpt_packing: no ready workers this round; keeping the previous assignment"
        else
            try
                rebalance!(p.packing, p.pool, ready_urls, p.metrics)
            catch e
                @warn "lpt_packing: rebalance failed; keeping the previous assignment" exception = e
            end
        end
    end
    return any_ready
end

# Wedge accounting: a round where EVERY probe missed even the watchdog (the clients' own 5s
# deadlines can only be missed when the libcurl multi has stopped being driven) means the
# client stack is dead and will not heal in-process. Exit so the supervisor / container restart
# policy brings up a fresh gateway, which reconnects immediately. Distinct from dead workers:
# those produce fast connection errors, not watchdog timeouts.
function _track_wedge!(p::HealthProber, all_timed_out::Bool)
    p.wedged_rounds = all_timed_out ? p.wedged_rounds + 1 : 0
    if p.wedge_exit_rounds > 0 && p.wedged_rounds >= p.wedge_exit_rounds
        @error "health: every ServerReady probe timed out for $(p.wedged_rounds) consecutive rounds; the gRPC client stack is wedged. Exiting so the supervisor restarts the gateway." rounds = p.wedged_rounds
        exit(1)
    end
    return nothing
end

# Probe once immediately, then on the interval until stopped.
function start_prober!(p::HealthProber)
    p.task = @async begin
        try
            _check_once(p)
        catch e
            @warn "health: initial probe failed" exception = e
        end
        while p.running[]
            sleep(p.interval)
            p.running[] || break
            try
                _check_once(p)
            catch e
                @warn "health: probe round failed" exception = e
            end
        end
    end
    return p
end

stop_prober!(p::HealthProber) = (p.running[] = false; nothing)
