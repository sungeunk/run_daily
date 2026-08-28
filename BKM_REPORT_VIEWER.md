# Legacy viewer (8501)

http://dg2raptorlake.ikor.intel.com:8501/

Old pickle/`.report`-based pipeline (`scripts/run_llm_daily.py`). Kept for
reference during the migration to the pytest-based `daily/` suite below.

## Settings
dg2raptorlake
src: /home/sungeunk/repo/run_daily/scripts/run_daily_report_viewer3.py
service file: /etc/systemd/system/viewer_daily_report.service
```bash
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
* service name is the filename.
sudo systemctl daemon-reload
sudo systemctl stop viewer_daily_report
sudo systemctl start viewer_daily_report
sudo systemctl status viewer_daily_report

sudo systemctl restart viewer_daily_report

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
```bash
[Unit]
 Description=Daily report viewer3

[Service]
 User=sungeunk
 WorkingDirectory=/home/sungeunk/repo/run_daily/scripts
 ExecStart=/home/sungeunk/miniforge3/envs/daily/bin/python -m streamlit run /home/sungeunk/repo/run_daily/daily/viewer/app.py -- --db /var/www/html/daily2/daily_llm_benchmark.duckdb
 Restart=always

[Install]
 WantedBy=multi-user.target
```

**Note:** currently `disabled` (won't auto-start on reboot) — only
`viewer_daily_report.service` (legacy, 8501) is `enabled`. Run
`sudo systemctl enable viewer_daily_report3` if this one should also survive
a reboot.

## Start service
* service name is the filename.
sudo systemctl daemon-reload
sudo systemctl stop viewer_daily_report3
sudo systemctl start viewer_daily_report3
sudo systemctl status viewer_daily_report3

sudo systemctl restart viewer_daily_report3

---

## Daily results MCP server (remote query via Copilot/Claude)

Toolkit source: `/home/sungeunk/repo/run_daily/daily/mcp_toolkit` (registered
locally with `gnai toolkits register`). Exposes 7 `daily_results_*` tools
over the `daily_llm_benchmark.duckdb` central DB. All tools are read-only
regardless of network/auth settings (DuckDB connection opened
`read_only=True`; the ad-hoc `daily_results_run_sql` tool rejects anything
but a single `SELECT`/`WITH` statement).

The companion skill (tool-selection guidance for "did it regress" / "show
trend" style questions) lives in the `openvino-gpu-plugin-skills` repo as
`query-daily-results` (`.github/skills/query-daily-results/SKILL.md`), not
in this toolkit.

- Local use (VS Code Copilot Chat / Claude Code on this machine): `.vscode/mcp.json`
  `command`-based entry, spawns `gnai toolkits serve daily_results --stdio` per session.
- Remote use: network-mode service below. GNAI auth stays enabled (teammates
  already have GNAI credentials — internal/anonymous requests get HTTP 401),
  SSL disabled since this is internal-network only.

service file: `/etc/systemd/system/daily_results_mcp.service`
```bash
[Unit]
Description=Daily results MCP server (gnai daily_results toolkit, network mode)
After=network.target

[Service]
User=sungeunk
WorkingDirectory=/home/sungeunk/repo/run_daily/daily/mcp_toolkit
ExecStart=/home/sungeunk/.config/dt/extensions/gnai/gnai toolkits serve daily_results --address=0.0.0.0:8090 --disable-ssl
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo cp /tmp/daily_results_mcp.service /etc/systemd/system/daily_results_mcp.service
sudo systemctl daemon-reload
sudo systemctl enable --now daily_results_mcp
sudo systemctl status daily_results_mcp
```

Remote teammate's `.vscode/mcp.json` (or Claude Code MCP config):
```json
{
  "servers": {
    "daily_results": {
      "url": "http://dg2raptorlake.ikor.intel.com:8090/mcp"
    }
  }
}
```
(Their `gnai` client will need to be logged in / have valid GNAI credentials
— the server answers HTTP 401 to unauthenticated requests.)

