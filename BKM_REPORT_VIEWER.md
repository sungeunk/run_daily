# Daily report services on dg2raptorlake

| Port | Service unit | What it is |
|------|--------------|------------|
| [8501](http://dg2raptorlake.ikor.intel.com:8501/) | `viewer_daily_report` | Legacy viewer (pickle/`.report` pipeline) |
| [8502](http://dg2raptorlake.ikor.intel.com:8502/) | `viewer_daily_report3` | Current viewer (pytest `daily/` pipeline) |
| [8090](http://dg2raptorlake.ikor.intel.com:8090/mcp) | `daily_results_mcp` | MCP server for agents (read-only, no auth) |

All three are `enabled` (auto-start on reboot). Service name == unit filename.

---

# Bootstrap (rebuilding this machine from scratch)

After a reinstall, only the `run_daily` repo is needed. The DuckDB file is
**not** restored — history starts empty and refills from the next ingest.

1. Miniforge + the `daily` conda env (every unit hardcodes this interpreter path):
   ```bash
   curl -L -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
   bash /tmp/miniforge.sh -b -p /home/sungeunk/miniforge3
   /home/sungeunk/miniforge3/bin/conda create -y -n daily python=3.11
   ```

2. Dependencies (covers the pytest suite, both Streamlit viewers and the MCP server):
   ```bash
   cd /home/sungeunk/repo/run_daily
   /home/sungeunk/miniforge3/envs/daily/bin/python -m pip install -r daily/requirements.txt
   ```

3. DB directory — the viewers and the MCP server all read
   `/var/www/html/daily2/daily_llm_benchmark.duckdb`:
   ```bash
   sudo mkdir -p /var/www/html/daily2
   sudo chown sungeunk:www-data /var/www/html/daily2
   sudo chmod 775 /var/www/html/daily2
   ```

4. Open the ports (ufw is active on this host):
   ```bash
   sudo ufw allow 8501/tcp
   sudo ufw allow 8502/tcp
   sudo ufw allow 8090/tcp
   ```

5. Register the three services from the sections below, then verify:
   ```bash
   for s in viewer_daily_report viewer_daily_report3 daily_results_mcp; do
     printf '%-24s %s / %s\n' "$s" "$(systemctl is-enabled $s)" "$(systemctl is-active $s)"
   done
   ```

---

# Legacy viewer (8501)

http://dg2raptorlake.ikor.intel.com:8501/

Old pickle/`.report`-based pipeline (`scripts/run_llm_daily.py`). Kept for
reference during the migration to the pytest-based `daily/` suite below.

## Settings
dg2raptorlake
src: /home/sungeunk/repo/run_daily/scripts/run_daily_report_viewer3.py
service file: /etc/systemd/system/viewer_daily_report.service
```ini
[Unit]
 Description=Daily report viewer

[Service]
 User=sungeunk
 WorkingDirectory=/home/sungeunk/repo/run_daily/scripts
 ExecStart=/home/sungeunk/miniforge3/envs/daily/bin/python -m streamlit run /home/sungeunk/repo/run_daily/scripts/run_daily_report_viewer3.py
 Restart=always

[Install]
 WantedBy=multi-user.target
```


## Start service
```bash
sudo systemctl daemon-reload
sudo systemctl stop viewer_daily_report
sudo systemctl start viewer_daily_report
sudo systemctl status viewer_daily_report

sudo systemctl restart viewer_daily_report
```

---

# Daily pipeline viewer — pytest-based (8502)

http://dg2raptorlake.ikor.intel.com:8502/

Current pipeline (`daily/` pytest suite, see `daily/README.md`). Reads the
same central DB the `daily_results` MCP tools query
(`/var/www/html/daily2/daily_llm_benchmark.duckdb`) — see `daily/viewer/README.md`
for the DuckDB schema and tab-by-tab breakdown (Excel/Trend/Regressions/Geomean/Noise).

## Settings
dg2raptorlake
src: /home/sungeunk/repo/run_daily/daily/viewer/app.py
service file: /etc/systemd/system/viewer_daily_report3.service
```ini
[Unit]
 Description=Daily report viewer3

[Service]
 User=sungeunk
 WorkingDirectory=/home/sungeunk/repo/run_daily/daily/viewer
 ExecStart=/home/sungeunk/miniforge3/envs/daily/bin/python -m streamlit run /home/sungeunk/repo/run_daily/daily/viewer/app.py -- --db /var/www/html/daily2/daily_llm_benchmark.duckdb
 Restart=always

[Install]
 WantedBy=multi-user.target
```

## Start service
```bash
sudo systemctl daemon-reload
sudo systemctl stop viewer_daily_report3
sudo systemctl start viewer_daily_report3
sudo systemctl status viewer_daily_report3

sudo systemctl restart viewer_daily_report3
```

---

# Daily results MCP server (8090)

http://dg2raptorlake.ikor.intel.com:8090/mcp

Remote query surface for Copilot/Claude.
Server source: `/home/sungeunk/repo/run_daily/daily/mcp_server/server.py` —
a standalone Python MCP server (official `mcp` SDK, `MCPServer`) exposing 7
`daily_results_*` tools over the `daily_llm_benchmark.duckdb` central DB.
It reuses `daily/viewer/queries.py`, the same query layer the Streamlit
viewer uses.

**No authentication.** Anyone on the internal network can query it. This is
safe because every tool is read-only: the DuckDB connection is opened
`read_only=True`, and the ad-hoc `daily_results_run_sql` tool additionally
rejects anything but a single `SELECT`/`WITH` statement (plus a keyword
denylist and an implicit `LIMIT 500`).

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

Dependencies live in the `daily` conda env — see Bootstrap above.

service file: `/etc/systemd/system/daily_results_mcp.service` — install with:
```bash
sudo tee /etc/systemd/system/daily_results_mcp.service > /dev/null <<'EOF'
[Unit]
Description=Daily results MCP server (standalone Python, no auth)
After=network.target

[Service]
User=sungeunk
WorkingDirectory=/home/sungeunk/repo/run_daily/daily/mcp_server
ExecStart=/home/sungeunk/miniforge3/envs/daily/bin/python /home/sungeunk/repo/run_daily/daily/mcp_server/server.py --transport streamable-http --host 0.0.0.0 --port 8090
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now daily_results_mcp
```

## Start service
```bash
sudo systemctl daemon-reload
sudo systemctl stop daily_results_mcp
sudo systemctl start daily_results_mcp
sudo systemctl status daily_results_mcp

sudo systemctl restart daily_results_mcp
```

Remote teammate's `.vscode/mcp.json` (or Claude Code MCP config) — no
credentials needed:
```json
{
  "servers": {
    "daily_results": {
      "url": "http://dg2raptorlake.ikor.intel.com:8090/mcp"
    }
  }
}
```

