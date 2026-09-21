# Web Server

This directory contains the user-level web services for `sungeunk`.

## Services

```text
web_server/
├── caddy/
├── files/
├── wiki/
└── update-wiki.sh
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
| `/daily/` | `/mnt/hdd/daily/` |
| `/benchmarking_datasets/` | `/mnt/hdd/jenkins/` |
| `/model_cache_server/` | `/mnt/hdd/model/` |

Local access:

```text
http://127.0.0.1:8081/daily/
http://127.0.0.1:8081/benchmarking_datasets/
http://127.0.0.1:8081/model_cache_server/
```

The file browser supports directory listings, downloads, and browser-native
viewing of formats such as text and HTML. It does not provide upload, delete,
authentication, or per-user access control.

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

Run the update script from this directory:

```bash
cd /home/sungeunk/repo/run_daily/web_server
./update-wiki.sh
```

The script performs these steps:

1. Runs MkDocs with Python 3.12 through `uv`.
2. Builds the site with `mkdocs build --strict`.
3. Reloads the user-level Caddy service only after a successful build.

To build the Wiki without reloading Caddy:

```bash
./update-wiki.sh --no-reload
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
