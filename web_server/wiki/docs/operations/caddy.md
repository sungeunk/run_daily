# Caddy

The generated Wiki site is served by the user-level Caddy service.

## Build and publish

From the `web_server` directory, run the service management script:

```bash
./manage-web-services.sh build-wiki
```

The script uses Python 3.12 through `uv`, builds with `--strict`, and reloads
Caddy only after a successful build.

Caddy serves the generated `site/` directory on port `8080`.

## Manage web services

The management script registers, enables, and operates Caddy, the Daily viewer,
and the Daily Results MCP server as user-level systemd services:

```bash
./manage-web-services.sh install
./manage-web-services.sh start
./manage-web-services.sh restart
./manage-web-services.sh status
```

`all` installs and restarts the services so updated unit settings take effect,
then builds the Wiki:

```bash
./manage-web-services.sh all
```

| Service | Endpoint | Backend |
|---|---|---|
| Caddy Wiki | `http://dg2fizz.ikor.intel.com:8080/` | `wiki/site/` |
| File browser | `http://dg2fizz.ikor.intel.com:8081/` | `/mnt/hdd/daily/data/` and other data roots |
| Daily Results MCP | `http://dg2fizz.ikor.intel.com:8090/mcp` | `/mnt/hdd/daily/db/daily_llm_benchmark.duckdb` |
| Daily viewer | `http://dg2fizz.ikor.intel.com:8091/` | Streamlit on `127.0.0.1:8501` |

## Check the services

```bash
systemctl --user is-active caddy.service
systemctl --user is-active daily-viewer.service
systemctl --user is-active daily-results-mcp.service
curl http://127.0.0.1:8080/
curl http://127.0.0.1:8501/_stcore/health
ss -ltn | grep ':8090'
```

The MCP endpoint requires a protocol initialization handshake, so a plain
browser request is not a complete functional test. Use an MCP client for tool
calls and inspect logs with:

```bash
journalctl --user -u daily-results-mcp.service -f
```
