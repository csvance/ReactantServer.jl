# Corporate / MITM root CA certificates

Drop additional trusted root CA certificates here when building on a network with a
TLS-inspecting proxy that re-signs HTTPS traffic. The Dockerfile copies this directory into
`/usr/local/share/ca-certificates/` and runs `update-ca-certificates`, and
`JULIA_SSL_CA_ROOTS_PATH` points Julia's NetworkOptions at the resulting system bundle so both
Downloads.jl (libcurl) and LibGit2 accept the proxy's certificates during `Pkg.instantiate`.

Requirements:
  - PEM-encoded (the `-----BEGIN CERTIFICATE-----` text format)
  - filename ending in `.crt` (update-ca-certificates ignores other extensions)

To capture the root your proxy presents:

    openssl s_client -showcerts -connect github.com:443 </dev/null 2>/dev/null \
      | awk '/BEGIN CERT/,/END CERT/'

The last certificate block in that output is the inspecting root. Save it here, e.g. as
`corp-root.crt`. With no cert present the build is unaffected (this README is not a `.crt`).
