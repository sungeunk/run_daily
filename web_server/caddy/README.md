# Caddy

User-level Caddy configuration for the `sungeunk` account. The setup does not require `sudo` and keeps the binary, service definition, and configuration under user-controlled paths.

## Repository layout

- `Caddyfile`: main configuration entry point. It imports every enabled file matching `conf.d/*.caddy`.
- `conf.d/*.caddy`: enabled site and reverse-proxy definitions.
- `conf.d/*.caddy.example`: disabled examples that are not imported.
- `systemd/caddy.service`: user service template.
- `~/.local/share/caddy`: Caddy certificates and runtime data, outside Git.
- `~/.local/state/caddy`: optional local state and logs, outside Git.

Do not commit certificates, private keys, credentials, or the Caddy binary.

## Install Caddy without sudo

The service expects the official Caddy binary at `~/.local/bin/caddy`.

1. Check the machine architecture:

	```bash
	uname -m
	```

2. Download the matching Linux binary from the official Caddy download endpoint. For an `x86_64` machine, for example:

	```bash
	set -eu
	mkdir -p "$HOME/.local/bin"
	tmp_dir="$(mktemp -d)"
	trap 'rm -rf "$tmp_dir"' EXIT
	curl --fail --location --silent --show-error \
		 'https://caddyserver.com/api/download?os=linux&arch=amd64' \
		 --output "$tmp_dir/caddy"
	install -m 755 "$tmp_dir/caddy" "$HOME/.local/bin/caddy"
	"$HOME/.local/bin/caddy" version
	```

	Use `arch=arm64` for an ARM64 machine. Verify the release and download details on the official Caddy site before installing.

3. Ensure the binary can be found in interactive shells:

	```bash
	export PATH="$HOME/.local/bin:$PATH"
	printf '\nexport PATH="$HOME/.local/bin:$PATH"\n' >> "$HOME/.bashrc"
	```

The systemd unit uses the absolute path `~/.local/bin/caddy`, so its operation does not depend on the service manager's `PATH`.

## Machine-specific configuration

Update these files for each machine before enabling the service:

### `Caddyfile`

- Keep `import conf.d/*.caddy` unless a different site layout is intentional.
- Change `admin localhost:2019` only when the local admin API port conflicts with another process. Do not expose the admin API publicly.

### `conf.d/*.caddy`

Create one file per site or reverse proxy. Start from the example:

```bash
cp conf.d/example.caddy.example conf.d/my-site.caddy
```

Update the following values:

- Site address or hostname, such as `:8080` or `app.example.com`.
- Listener port. Without `sudo`, use a port above 1024, such as `8080`.
- `reverse_proxy` upstream address and port for the application server.
- TLS settings, DNS challenge configuration, and contact email when the machine handles HTTPS.
- Machine-specific file paths or environment variables. Keep secrets outside Git.

Only files ending in `.caddy` are active. Files ending in `.caddy.example` and `.local.caddy` are ignored by the import pattern or `.gitignore`.

### `systemd/caddy.service`

The unit currently assumes:

- User: `sungeunk`.
- Repository: `/home/sungeunk/repo/run_daily/web_server/caddy`.
- Binary: `/home/sungeunk/.local/bin/caddy`.

If the user, repository location, or binary location changes, update `ExecStart` and `ExecReload` before installing the unit.

## Enable and start the user service

From this directory, install the tracked unit and enable it for the `sungeunk` login session:

```bash
mkdir -p "$HOME/.config/systemd/user"
cp systemd/caddy.service "$HOME/.config/systemd/user/caddy.service"
systemctl --user daemon-reload
systemctl --user enable --now caddy.service
systemctl --user status caddy.service
```

To make the service start when the user is not logged in, system-wide lingering must be enabled. That normally requires an administrator:

```bash
loginctl enable-linger sungeunk
```

Without lingering, the service runs while the user session is active.

## Validate, reload, and troubleshoot

Validate before applying a configuration change:

```bash
~/.local/bin/caddy validate \
	 --config "$HOME/repo/run_daily/web_server/caddy/Caddyfile" \
	 --adapter caddyfile
```

Reload the service after validation:

```bash
systemctl --user reload caddy.service
```

Useful status and log commands:

```bash
systemctl --user is-enabled caddy.service
systemctl --user is-active caddy.service
journalctl --user -u caddy.service -e
journalctl --user -u caddy.service -f
```

The service runs as `sungeunk` and cannot bind privileged ports such as 80 or 443 without additional administrator-managed port forwarding or a separate reverse proxy.
