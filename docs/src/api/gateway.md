```@meta
CurrentModule = ReactantServerGateway
```

# Gateway

`ReactantServerGateway` is the KServe V2 gRPC reverse proxy that fronts the workers of a
multi-GPU node. The node supervisor runs it as an embedded child when there is more than one
worker. It depends only on `ReactantServerCore` and the gRPC/HTTP layer, never on Reactant. See
[Multi-GPU Gateway](../manual/multi_gpu_gateway.md) for the operational view.

```@docs
serve_gateway
probe_worker_ready
RunningGateway
stop!
```
