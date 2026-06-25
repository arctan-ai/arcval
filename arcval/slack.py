"""Slack webhook notifications for arcval runs."""

import json
import mimetypes
import os
import uuid
import urllib.request

WEBHOOK_ENV_VAR = "SLACK_WEBHOOK_URL"
BOT_TOKEN_ENV_VAR = "SLACK_BOT_TOKEN"
CHANNEL_ID_ENV_VAR = "SLACK_CHANNEL_ID"


def webhook_url() -> str | None:
    """Return the Slack webhook URL from the environment, or None."""
    return os.environ.get(WEBHOOK_ENV_VAR)


def bot_token() -> str | None:
    """Return the Slack bot token from the environment, or None."""
    return os.environ.get(BOT_TOKEN_ENV_VAR)


def channel_id() -> str | None:
    """Return the Slack channel ID from the environment, or None."""
    return os.environ.get(CHANNEL_ID_ENV_VAR)


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


def _encode_multipart_form(fields: dict[str, str]) -> tuple[bytes, str]:
    """Encode a small multipart/form-data payload for Slack form endpoints."""
    boundary = f"----ArcvalSlackBoundary{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode(),
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), boundary


def _slack_api_form_call(
    endpoint: str,
    fields: dict[str, str],
    token: str,
) -> dict:
    """Call a Slack Web API form endpoint and return the parsed JSON body."""
    body, boundary = _encode_multipart_form(fields)
    req = urllib.request.Request(
        f"https://slack.com/api/{endpoint}",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        payload = json.loads(response.read().decode())
    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", f"Slack API call failed: {endpoint}"))
    return payload


def upload_file(
    file_path: str,
    initial_comment: str,
    title: str | None = None,
    token: str | None = None,
    channel: str | None = None,
) -> None:
    """Upload a file to Slack and share it in a channel."""
    token = token or bot_token()
    channel = channel or channel_id()
    if not token or not channel:
        raise RuntimeError("Slack file upload requires SLACK_BOT_TOKEN and SLACK_CHANNEL_ID")

    filename = os.path.basename(file_path)
    size = os.path.getsize(file_path)
    upload_meta = _slack_api_form_call(
        "files.getUploadURLExternal",
        {
            "filename": filename,
            "length": str(size),
        },
        token,
    )

    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    with open(file_path, "rb") as handle:
        upload_req = urllib.request.Request(
            upload_meta["upload_url"],
            data=handle.read(),
            headers={"Content-Type": content_type},
            method="POST",
        )
    with urllib.request.urlopen(upload_req, timeout=60) as response:
        response.read()

    _slack_api_form_call(
        "files.completeUploadExternal",
        {
            "files": json.dumps(
                [{"id": upload_meta["file_id"], "title": title or filename}]
            ),
            "channel_id": channel,
            "initial_comment": initial_comment,
        },
        token,
    )
