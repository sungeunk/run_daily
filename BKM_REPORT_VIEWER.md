# Daily report services on dg2fizz

| Port | Service unit | What it is |
|------|--------------|------------|
| [8080](http://dg2fizz.ikor.intel.com:8080/) | `caddy.service` | Wiki |
| [8081](http://dg2fizz.ikor.intel.com:8081/) | `caddy.service` | Result file browser |
| [8091](http://dg2fizz.ikor.intel.com:8091/) | `daily-viewer.service` | Current viewer (pytest `daily/` pipeline) |
| [8090](http://dg2fizz.ikor.intel.com:8090/mcp) | `daily-results-mcp.service` | MCP server for agents (read-only, no auth) |

All listed user services are enabled and start automatically through user
lingering. Service names match their tracked unit filenames.

---

# Bootstrap (rebuilding this machine from scratch)

After a reinstall, only the `run_daily` repo is needed. The DuckDB file is
**not** restored — history starts empty and refills from the next ingest.

1. Install `uv`:
   ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  sudo ln -sf "$HOME/.local/bin/uv" /usr/local/bin/uv
   ```

  Install Caddy at `/home/sungeunk/.local/bin/caddy` as documented in
  `web_server/caddy/README.md`.

2. Verify the tracked requirements through `uv`:
   ```bash
   cd /home/sungeunk/repo/run_daily
  uv run --with-requirements daily/requirements.txt python -c "import duckdb, streamlit"
   ```

3. DB directory — the viewers and the MCP server all read
  `/mnt/hdd/daily/db/daily_llm_benchmark.duckdb`:
   ```bash
  sudo mkdir -p /mnt/hdd/daily/db
  sudo chown sungeunk:devel /mnt/hdd/daily/db
  sudo chmod 755 /mnt/hdd/daily/db
   ```

4. Open the ports (ufw is active on this host):
   ```bash
    sudo ufw allow 8080/tcp
    sudo ufw allow 8081/tcp
    sudo ufw allow 8090/tcp
    sudo ufw allow 8091/tcp
   ```

5. Register, enable, and start the tracked user services:
   ```bash
  cd /home/sungeunk/repo/run_daily/web_server
  ./manage-web-services.sh all
  ./manage-web-services.sh status
   ```

  Enable user lingering once so services start after reboot without an
  interactive login:

  ```bash
  sudo loginctl enable-linger sungeunk
  ```

**Still missing after these steps**:

- The benchmark side itself (OpenVINO + GenAI runtime, models, prompts) if
  this host is also meant to *run* `daily/` pytest, not just serve results.
  `daily/requirements.txt` only covers the viewers, the MCP server and the
  test harness — not OpenVINO/GenAI.
- There is no crontab entry on this host; whatever triggers the daily runs and
  the ingest lives elsewhere and has to be reconnected.

---

# Legacy viewer — retired on the old host

The pickle/`.report`-based pipeline and its viewer were deleted once the
`daily/` pytest suite covered every tab it offered (Excel Paste and
Summary & Chart map onto Excel and Dashboard/Compare). Recover from git
history if a question about a pre-migration run ever needs it:

```bash
git log --diff-filter=D -- scripts/run_daily_report_viewer3.py
```

The old service and `/var/www/html/daily/` artifacts belonged to the retired
host and are not part of the `dg2fizz` deployment.

---

# Daily pipeline viewer — pytest-based (8091)

http://dg2fizz.ikor.intel.com:8091/

Current pipeline (`daily/` pytest suite, see `daily/README.md`). Reads the
same central DB the `daily_results` MCP tools query
(`/mnt/hdd/daily/db/daily_llm_benchmark.duckdb`) — see `daily/viewer/README.md`
for the DuckDB schema and tab-by-tab breakdown (Excel/Trend/Regressions/Geomean/Noise).

## Settings
dg2fizz
src: /home/sungeunk/repo/run_daily/daily/viewer/app.py
service file: /home/sungeunk/repo/run_daily/web_server/caddy/systemd/daily-viewer.service
```ini
[Unit]
Description=Daily Streamlit viewer (sungeunk user)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=%h/repo/run_daily
Environment=DAILY_DB=/mnt/hdd/daily/db/daily_llm_benchmark.duckdb
Environment=INGEST_SCRIPT=/home/sungeunk/repo/run_daily/scripts/ingest_db.sh
Environment=INGEST_LOCK_FILE=/mnt/hdd/daily/db/.ingest.lock
Environment=STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ExecStart=/usr/local/bin/uv run --python 3.12 --with-requirements %h/repo/run_daily/daily/requirements.txt streamlit run %h/repo/run_daily/daily/viewer/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=default.target
```

## Start service
```bash
cd /home/sungeunk/repo/run_daily/web_server
./manage-web-services.sh install
systemctl --user restart daily-viewer.service
systemctl --user status daily-viewer.service
```

---

# Daily results MCP server (8090)

http://dg2fizz.ikor.intel.com:8090/mcp

Remote query surface for Copilot/Claude.
Server source: `/home/sungeunk/repo/run_daily/daily/mcp_server/server.py` —
a standalone Python MCP server (official `mcp` SDK, `MCPServer`) exposing 7
`daily_results_*` tools over the `daily_llm_benchmark.duckdb` central DB.
It reuses `daily/data/read.py`, the same query layer the Streamlit viewer uses.

**No authentication.** Anyone on the internal network can query it, so the
server is hardened rather than trusted: the DuckDB connection is opened
`read_only=True` **and** with `enable_external_access=False`, which is what
stops `read_text`/`read_csv_auto` from reading arbitrary host files. On top of
that `daily_results_run_sql` accepts only a single `SELECT`/`WITH` statement
(string literals are blanked before the keyword denylist runs, so `;` or
`DELETE` inside a literal neither bypasses nor trips the check) and every
query is capped at 500 rows when fetched, independent of the SQL text.

> Replaced the earlier `gnai toolkits serve` deployment (`daily/mcp_toolkit`).
> gnai was only providing the MCP transport, the tool schemas and a venv for
> the per-call subprocesses — none of which is needed now that the tools are
> plain Python functions in one process. Clients no longer need `gnai`
> installed or a GNAI login.

The companion skill (tool-selection guidance for "did it regress" / "show
trend" style questions) lives in the `openvino-gpu-plugin-skills` repo as
`query-daily-results` (`.github/skills/query-daily-results/SKILL.md`).

- Local use (VS Code Copilot Chat / Claude Code on this machine):
  `.vscode/mcp.json` spawns `server.py --transport stdio` per session.
- Remote use: the network-mode service below (plain HTTP, internal network only).

Dependencies are resolved by `uv` from `daily/requirements.txt`.

The tracked user service is
`web_server/caddy/systemd/daily-results-mcp.service`. Register and start it
through the shared service manager:

```bash
cd /home/sungeunk/repo/run_daily/web_server
./manage-web-services.sh install
./manage-web-services.sh start
```

## Start service
```bash
systemctl --user restart daily-results-mcp.service
systemctl --user status daily-results-mcp.service
journalctl --user -u daily-results-mcp.service -f
```

Remote teammate's `.vscode/mcp.json` (or Claude Code MCP config) — no
credentials needed:
```json
{
  "servers": {
    "daily_results": {
      "url": "http://dg2fizz.ikor.intel.com:8090/mcp"
    }
  }
}
```

