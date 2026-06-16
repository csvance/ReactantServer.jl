# Contributing

Developer notes for working on ReactantServer.jl. For using the server, start with
[Getting Started](docs/src/manual/getting_started.md).

After cloning, populate the vendored submodules and instantiate the workspace:

```
git submodule update --init --recursive
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Testing

Each package is tested in its own environment; all tests run on CPU and need no GPU:

```
julia --project=packages/ReactantServerCore    -e 'using Pkg; Pkg.test()'
julia --project=packages/ReactantServer         -e 'using Pkg; Pkg.test()'
julia --project=packages/ReactantServerGateway  -e 'using Pkg; Pkg.test()'
julia --project=packages/ReactantServerClient   -e 'using Pkg; Pkg.test()'
julia --project=packages/ReactantServerNode     -e 'using Pkg; Pkg.test()'
```

The worker's pre/post-processing overlaps GPU execution only with more than one thread, so run its
suite multithreaded to exercise that path: `Pkg.test(; julia_args=["--threads=auto,1"])`.

`ReactantServerExport` is not a workspace member; its export round-trip tests (export a model, run
it through the runtime, and compare to a native Lux/PyTorch forward pass) run under their own env:

```
julia --project=packages/ReactantServerExport/test packages/ReactantServerExport/test/runtests.jl
```

The PyTorch portion skips gracefully when `torch`/`torchax` are unavailable.
`packages/ReactantServer/test/spike_reactant.jl` (and the `spike_*.jl` siblings) are standalone
scripts that exercise the Reactant runtime and export paths in isolation.

## Regenerating the protobuf bindings

The KServe V2 messages and gRPC service stubs in
`packages/ReactantServerCore/src/proto/inference/` are generated from
`proto_src/grpc_predict_v2.proto` with ProtoBuf.jl. Load gRPCServer and gRPCClient alongside
ProtoBuf so both the server method builders / `register_GRPCInferenceService!` and the client
constructors are emitted, and keep `add_kwarg_constructors=true` (the handlers and codec build
messages with keyword arguments):

```julia
using ProtoBuf, gRPCServer, gRPCClient
ProtoBuf.protojl("grpc_predict_v2.proto", "proto_src", "packages/ReactantServerCore/src/proto";
    always_use_modules=true, add_kwarg_constructors=true)
```

The generated file is then split so `ReactantServerCore` compiles only the messages (no gRPC
dependency): the messages stay in `grpc_predict_v2_pb.jl`, while the gRPCClient and gRPCServer
service stubs are extracted into `grpc_client_stubs.jl` and `grpc_server_stubs.jl`, which
`ReactantServerCore` ships but does not compile. Each consumer includes the stub file it needs
(client stubs in the client and gateway; server stubs in the worker and gateway) via
`ReactantServerCore.inference_client_stubs_path()` / `inference_server_stubs_path()`.

## Documentation

The docs are built with Documenter:

```
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Output lands in `docs/build/`. There is no `deploydocs`; an Azure DevOps pipeline can publish
`docs/build/` as an artifact when desired.
