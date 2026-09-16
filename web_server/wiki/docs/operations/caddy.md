# Caddy

The generated Wiki site is served by the user-level Caddy service.

## Build and publish

From the `web_server` directory, run the Wiki update script:

```bash
./update-wiki.sh
```

The script uses Python 3.12 through `uv`, builds with `--strict`, and reloads
Caddy only after a successful build. Use `--no-reload` to build without
reloading the service.

Caddy serves the generated `site/` directory on port `8080`.

## Check the service

```bash
systemctl --user is-active caddy.service
curl http://127.0.0.1:8080/
```
