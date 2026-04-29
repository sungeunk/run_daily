import argparse
import json
import os
import sys
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("status", choices=STATUS.keys())
    args = parser.parse_args()

    status = args.status

    webhook = os.environ["WEBHOOK_TO_TEAMS_CHAT"]
    card = build_card(status)

    req = urllib.request.Request(
        webhook,
        data=json.dumps(card).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        print(f"sent [{status}]: {resp.status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
