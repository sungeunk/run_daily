# Web Server

This directory contains the user-level web services for `sungeunk`.

## Services

```text
web_server/
├── caddy/
├── files/
├── wiki/
└── manage-web-services.sh
```

### Caddy

Caddy is the front-end web server and runs as the `sungeunk` user through a
`systemd --user` service. Its main configuration is
`caddy/Caddyfile`, which imports the site configurations from
`caddy/conf.d/`.

See `caddy/README.md` for Caddy installation, service enablement, validation,
and troubleshooting details.

The Wiki site is configured in `caddy/conf.d/wiki.caddy`:

```text
Browser -> Caddy :8080 -> wiki/site/
```

Caddy serves the generated static files directly. Wiki.js or another
application server is not required.

### Wiki

The Wiki source files are Markdown under `wiki/docs/`. MkDocs with the
Material theme converts them into the static site under `wiki/site/`.

- Source documents: `wiki/docs/`
- MkDocs configuration: `wiki/mkdocs.yml`
- Python dependencies: `wiki/requirements.txt`
- Generated site: `wiki/site/`
- Generated site is excluded from Git

The Wiki build uses Python 3.12 and `uv`. It does not require a checked-in
virtual environment or a system-wide MkDocs installation.

### File browser

Caddy also provides read-only directory browsing on port `8081`. The mappings
are defined in `caddy/conf.d/files.caddy`.

| URL | Directory |
| --- | --- |
| `/daily/` | `/mnt/hdd/daily/data/` |
| `/daily2/` | `/mnt/hdd/daily/data/` |
| `/benchmarking_datasets/` | `/mnt/hdd/jenkins/` |
| `/model_cache_server/` | `/mnt/hdd/model/` |

Local access:

```text
http://127.0.0.1:8081/daily/
http://127.0.0.1:8081/benchmarking_datasets/
http://127.0.0.1:8081/model_cache_server/
```

The file browser supports directory listings, downloads, and browser-native
viewing of formats such as text and HTML. `.raw` files are served as
`text/plain` and open directly in the browser. `.parquet` files are binary
files, so they are served as downloads rather than rendered as text. The file
browser does not provide upload, delete, authentication, or per-user access
control.

### Daily viewer

The daily viewer runs on `dg2fizz` as a user-level Streamlit service. The
Jenkins controller remains on `dg2ubuntu`; this service only hosts the viewer
and reads the local daily data on `dg2fizz`.

```text
Browser -> Caddy :8091 -> Streamlit 127.0.0.1:8501
                           -> /mnt/hdd/daily/db/daily_llm_benchmark.duckdb
```

The service unit is
`caddy/systemd/daily-viewer.service`. It uses `uv` and the dependencies in
`daily/requirements.txt`, and sets `DAILY_DB` explicitly so the viewer does
not depend on the machine hostname.

Install and start it as the `sungeunk` user:

```bash
./manage-web-services.sh install
systemctl --user start daily-viewer.service
systemctl --user status daily-viewer.service
```

Caddy exposes the viewer at `http://dg2fizz.ikor.intel.com:8091/` and proxies
to the private Streamlit port `8501`. Reload Caddy after installing the route:

```bash
systemctl --user reload caddy.service
```

Useful checks:

```bash
curl http://127.0.0.1:8501/_stcore/health
journalctl --user -u daily-viewer.service -f
```

Daily result artifacts are stored under `/mnt/hdd/daily/data/`, while the
database and its backups are stored under `/mnt/hdd/daily/db/`. The service
uses `/mnt/hdd/daily/db/daily_llm_benchmark.duckdb` as `DAILY_DB`; update the
unit if the ingestion job changes the filename. The service is intentionally
limited to the `sungeunk` user and does not expose Streamlit directly on the
network.

### Daily results MCP

The read-only MCP server runs as `daily-results-mcp.service` and exposes the
same daily result database used by the Streamlit viewer:

```text
MCP client -> http://dg2fizz.ikor.intel.com:8090/mcp
              -> /mnt/hdd/daily/db/daily_llm_benchmark.duckdb
```

The service unit is
`caddy/systemd/daily-results-mcp.service`. It uses `uv`, binds to port `8090`,
and opens DuckDB read-only. It is registered and managed by the same script:

```bash
./manage-web-services.sh install
./manage-web-services.sh start
./manage-web-services.sh restart
./manage-web-services.sh status
```

Check the MCP service with:

```bash
systemctl --user status daily-results-mcp.service
journalctl --user -u daily-results-mcp.service -f
curl -i http://127.0.0.1:8090/mcp
```

The endpoint has no authentication and should only be reachable from the
trusted internal network.

### Jenkins node

The `dg2fizz` machine can connect to the Jenkins controller as the `dg2fizz`
agent through a user-level `systemd` service. The agent files are stored in
`/home/sungeunk/jenkins/`:

- `agent.jar`: Jenkins Remoting agent downloaded from the controller
- `secret-file`: agent secret; keep this file private
- `remoting/`: agent work directory and logs

Create the user service unit at
`~/.config/systemd/user/jenkins-agent.service`:

```ini
[Unit]
Description=Jenkins Agent dg2fizz
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/sungeunk/jenkins
ExecStart=/usr/bin/java -jar /home/sungeunk/jenkins/agent.jar -url http://dg2ubuntu.ikor.intel.com:8080/ -secret @/home/sungeunk/jenkins/secret-file -name dg2fizz -webSocket -workDir /home/sungeunk/jenkins
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
```

The service requires a JDK with the `java` executable available. Verify the
path before creating the unit:

```bash
command -v java
java -version
```

Make sure the agent files belong to `sungeunk`, and restrict the secret file:

```bash
chown sungeunk:sungeunk /home/sungeunk/jenkins/agent.jar /home/sungeunk/jenkins/secret-file
chmod 600 /home/sungeunk/jenkins/secret-file
```

Enable and start the service:

```bash
systemctl --user daemon-reload
systemctl --user enable --now jenkins-agent.service
systemctl --user status jenkins-agent.service
```

Follow the service log with:

```bash
journalctl --user -u jenkins-agent.service -f
```

To keep the user service running after logout and start it after reboot, enable
lingering once with administrator privileges:

```bash
sudo loginctl enable-linger sungeunk
```

Successful startup includes `WebSocket connection open` and `Connected` in the
service log. If `agent.jar` contains an HTML `Access Denied` page instead of a
Java archive, download it from a network path that can reach the internal
Jenkins controller and verify it before starting the service.

## Update the Wiki

Run the service management script from this directory:

```bash
cd /home/sungeunk/repo/run_daily/web_server
./manage-web-services.sh build-wiki
```

The script also manages the user-level service units:

```bash
./manage-web-services.sh install
./manage-web-services.sh start
./manage-web-services.sh restart
./manage-web-services.sh status
```

`install` registers and enables available Caddy, Daily viewer, and Daily
Results MCP units so they start with the user systemd manager after reboot.
The `all` command installs services, restarts them so updated unit settings take
effect, builds the Wiki, and reloads Caddy:

```bash
./manage-web-services.sh all
```

The script runs MkDocs with Python 3.12 through `uv` and reloads Caddy only
after a successful build. Service logs can be followed with:

```bash
journalctl --user -u caddy.service -f
journalctl --user -u daily-viewer.service -f
```

If the build fails, Caddy is not reloaded and the previously published site
remains available.

## Access

Local services:

```text
Wiki:         http://127.0.0.1:8080/
File browser: http://127.0.0.1:8081/
```

These services use unprivileged ports because Caddy runs as the `sungeunk` user.
Access from another machine requires network reachability and firewall
permission for the relevant port. Ports `80` and `443` require
administrator-managed port forwarding or a separate privileged reverse proxy.

## Git management

Commit the Markdown source, MkDocs configuration, Caddy configuration, and
update script. Do not commit generated files, runtime data, certificates,
private keys, or local secrets.
