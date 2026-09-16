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
are defined in `caddy/conf.d/files.caddy`:

| URL | Directory |
| --- | --- |
| `/daily/` | `/mnt/hdd/daily/` |
| `/benchmarking_datasets/` | `/mnt/hdd/jenkins/` |
| `/model_cache_server/` | `/mnt/hdd/model/` |

For example, on the local machine:

```text
http://127.0.0.1:8081/daily/
http://127.0.0.1:8081/benchmarking_datasets/
http://127.0.0.1:8081/model_cache_server/
```

The file browser supports directory listings, downloads, and browser-native
viewing of formats such as text and HTML. It does not provide upload, delete,
authentication, or per-user access control.

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

## Service operations

Check Caddy:

```bash
systemctl --user is-active caddy.service
systemctl --user status caddy.service
```

Validate the Caddy configuration:

```bash
~/.local/bin/caddy validate \
    --config /home/sungeunk/repo/run_daily/web_server/caddy/Caddyfile \
    --adapter caddyfile
```

Reload Caddy manually when needed:

```bash
systemctl --user reload caddy.service
```

## Access

The Wiki is currently available locally at:

```text
http://127.0.0.1:8080/
```

The service uses an unprivileged port because it runs without `sudo`. Access
from another machine requires network reachability and firewall permission for
port `8080`. Ports `80` and `443` require administrator-managed port
forwarding or a separate privileged reverse proxy.

## Git management

Commit the Markdown source, MkDocs configuration, Caddy configuration, and
update script. Do not commit generated files, runtime data, certificates,
private keys, or local secrets.
