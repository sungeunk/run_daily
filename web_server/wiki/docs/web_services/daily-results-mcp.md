# Daily Results MCP

The Daily Results MCP server exposes read-only tools for querying the central
benchmark database on `dg2fizz`.

| Item | Value |
|---|---|
| Endpoint | `http://dg2fizz.ikor.intel.com:8090/mcp` |
| User service | `daily-results-mcp.service` |
| Database | `/mnt/hdd/daily/db/daily_llm_benchmark.duckdb` |
| Source | `daily/mcp_server/server.py` |

## Service management

From `web_server/`:

```bash
./manage-web-services.sh install
./manage-web-services.sh start
./manage-web-services.sh restart
./manage-web-services.sh status
```

Direct checks:

```bash
systemctl --user status daily-results-mcp.service
journalctl --user -u daily-results-mcp.service -f
ss -ltn | grep ':8090'
```

The MCP endpoint requires an MCP initialization handshake; a plain browser or
uninitialized HTTP tool request is not a valid functional check.

## Security

The endpoint has no authentication. DuckDB is opened read-only and arbitrary
SQL is restricted to a single `SELECT` or `WITH` statement. Keep port `8090`
limited to the trusted internal network.
