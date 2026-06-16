#!/usr/bin/env bash
# Launch the node supervisor: detect the visible GPUs, spawn one ReactantServer worker
# subprocess per device plus the embedded gateway (role `all`, the default), multiplex their
# output onto this container's stdout with [name] line prefixes, and restart children that die.
# Roles (REACTANT_ROLE): all | workers | gateway. The node file (REACTANT_NODE_FILE, default
# /etc/reactantserver/node.yaml) needs no workers list; one is synthesized per detected GPU.
set -euo pipefail

# --handle-signals=no lets the supervisor's own SIGTERM/SIGINT handler run (Julia's default
# runtime handling would otherwise consume SIGTERM and die without shutting children down).
exec julia --handle-signals=no --project=/opt/reactantserver/packages/ReactantServerNode -e '
    using ReactantServerNode
    ReactantServerNode.main()
'
