"""Slack webhook notifications for arcval runs."""

import json
import os
import urllib.request

WEBHOOK_ENV_VAR = "SLACK_WEBHOOK_URL"


def webhook_url() -> str | None:
    """Return the Slack webhook URL from the environment, or None."""
    return os.environ.get(WEBHOOK_ENV_VAR)


def send_message(text: str, url: str | None = None) -> None:
    """Send a plain-text Slack message via Incoming Webhook.

    Args:
        text: Message text (supports Slack mrkdwn formatting).
        url: Webhook URL. Falls back to ``SLACK_WEBHOOK_URL`` env var.
    """
    url = url or webhook_url()
    if not url:
        return
    try:
        data = json.dumps({"text": text}).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass
