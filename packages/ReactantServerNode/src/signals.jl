# SIGTERM/SIGINT bridge. The C handler is restricted to uv_async_send (async-signal-safe); the
# Julia side waits on the AsyncCondition and runs the graceful shutdown from a normal task.
#
# IMPORTANT: this only works when Julia is started with `--handle-signals=no`. With the default
# signal handling, the runtime consumes SIGTERM on its own signal thread (printing a crash
# report and exiting 143) before any user handler is consulted; signal()/sigaction dispositions
# are never reached. The supervisor is a light single-threaded orchestrator (no Reactant, @async
# only), so disabling the runtime handlers is safe; the entrypoint passes the flag. Children set
# PR_SET_PDEATHSIG as the kernel-level backstop, so even an ungraceful supervisor death can not
# orphan them.

const _SHUTDOWN_COND = Ref{Base.AsyncCondition}()

function _on_signal(::Cint)
    # Only uv_async_send here: a signal handler may run on any thread at any point, so no
    # allocation, locks, or Julia runtime entry.
    ccall(:uv_async_send, Cint, (Ptr{Cvoid},), _SHUTDOWN_COND[].handle)
    return nothing
end

"""
    install_term_handlers!(cond) -> nothing

Route SIGTERM and SIGINT to `cond` (a `Base.AsyncCondition`). Effective only under
`julia --handle-signals=no` (see the file comment); a warning is emitted otherwise. Tests drive
shutdown through `request_shutdown!` instead and never install these.
"""
function install_term_handlers!(cond::Base.AsyncCondition)
    if Base.JLOptions().handle_signals == 1
        @warn "Julia's default signal handling is active: SIGTERM will terminate the supervisor without graceful child shutdown (children still exit via PR_SET_PDEATHSIG). Start the supervisor with julia --handle-signals=no for graceful shutdown."
    end
    _SHUTDOWN_COND[] = cond
    h = @cfunction(_on_signal, Cvoid, (Cint,))
    ccall(:signal, Ptr{Cvoid}, (Cint, Ptr{Cvoid}), 15, h)   # SIGTERM
    ccall(:signal, Ptr{Cvoid}, (Cint, Ptr{Cvoid}), 2, h)    # SIGINT
    return nothing
end
