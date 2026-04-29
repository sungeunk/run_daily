import argparse
import json
import os
import ssl
import sys
import time
import urllib.request

STATUS = {
    "start": {"icon": "🚀", "keyword": "STARTED",  "color": "Accent"},
    "end":   {"icon": "🏁", "keyword": "FINISHED", "color": "Default"},
}


def build_card(status: str) -> dict:
    s = STATUS[status]
    name = os.environ.get("BUILD_DISPLAY_NAME", "-")
    url  = os.environ.get("BUILD_URL", "https://example.com")

    return {
        "type": "AdaptiveCard",
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "version": "1.4",
        "body": [
            {
                "type": "TextBlock",
                "size": "Medium",
                "weight": "Bolder",
                "color": s["color"],
                "text": f"{s['icon']} {s['keyword']} - {name}",
            },
        ],
        "actions": [
            {"type": "Action.OpenUrl", "title": "Open build", "url": url},
        ],
    }


def send_teams_message(webhook: str, card: dict, max_retries: int = 3) -> bool:
    """Send Teams message with retry logic for SSL errors."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                webhook,
                data=json.dumps(card).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            # Create SSL context that's more tolerant of corporate proxies
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                print(f"sent: {resp.status}")
                return True

        except (ssl.SSLError, urllib.error.URLError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
                return False
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("status", choices=STATUS.keys())
    args = parser.parse_args()

    status = args.status
    webhook = os.environ["WEBHOOK_TO_TEAMS_CHAT"]
    card = build_card(status)

    success = send_teams_message(webhook, card)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
