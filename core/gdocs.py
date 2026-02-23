"""Google Docs API integration.

Requires valid OAuth2 credentials at config/google_oauth.json.
The user token is cached at ~/.config/veritas_reader/token.json.
"""

import logging
from pathlib import Path

from platformdirs import user_config_dir

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(user_config_dir("veritas_reader"))
_TOKEN_PATH = _CONFIG_DIR / "token.json"
_CREDS_PATH = Path(__file__).parent.parent / "config" / "google_oauth.json"

SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive.file",
]


class GDocsError(Exception):
    """Raised when Google Docs operations fail."""


def _get_credentials():
    """Load or refresh OAuth2 credentials, triggering browser flow if needed."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise GDocsError(
            "Google API libraries not installed. Run: "
            "pip install google-api-python-client google-auth-oauthlib"
        ) from exc

    creds = None
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if _TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(_TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not _CREDS_PATH.exists():
                raise GDocsError(
                    f"Google OAuth credentials not found at {_CREDS_PATH}. "
                    "Download credentials.json from Google Cloud Console and "
                    "place it at config/google_oauth.json."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(_CREDS_PATH), SCOPES)
            creds = flow.run_local_server(port=0)

        _TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
        logger.info("Google credentials saved to %s", _TOKEN_PATH)

    return creds


def _build_service():
    from googleapiclient.discovery import build
    creds = _get_credentials()
    return build("docs", "v1", credentials=creds)


def authenticate() -> bool:
    """Trigger the OAuth flow and cache credentials.

    Returns:
        True on success, raises GDocsError on failure.
    """
    _get_credentials()
    return True


def read_doc(doc_id: str) -> str:
    """Fetch the plain text content of a Google Doc.

    Args:
        doc_id: The document ID from the URL:
                https://docs.google.com/document/d/<doc_id>/edit

    Returns:
        The document content as a plain text string.
    """
    try:
        service = _build_service()
        doc = service.documents().get(documentId=doc_id).execute()
        return _extract_text(doc)
    except Exception as exc:
        raise GDocsError(f"Failed to read document {doc_id}: {exc}") from exc


def write_doc(doc_id: str, text: str) -> None:
    """Replace all content in a Google Doc with the given text.

    Args:
        doc_id: Target document ID.
        text: New plain text content.
    """
    try:
        service = _build_service()
        doc = service.documents().get(documentId=doc_id).execute()
        end_index = doc["body"]["content"][-1]["endIndex"] - 1

        requests_body = []
        if end_index > 1:
            requests_body.append({
                "deleteContentRange": {
                    "range": {"startIndex": 1, "endIndex": end_index}
                }
            })
        requests_body.append({
            "insertText": {"location": {"index": 1}, "text": text}
        })

        service.documents().batchUpdate(
            documentId=doc_id, body={"requests": requests_body}
        ).execute()
        logger.info("Updated Google Doc %s", doc_id)
    except Exception as exc:
        raise GDocsError(f"Failed to write document {doc_id}: {exc}") from exc


def create_doc(title: str, text: str) -> str:
    """Create a new Google Doc and populate it with text.

    Args:
        title: Document title.
        text: Initial content.

    Returns:
        The new document's ID.
    """
    try:
        service = _build_service()
        doc = service.documents().create(body={"title": title}).execute()
        doc_id = doc["documentId"]
        if text:
            service.documents().batchUpdate(
                documentId=doc_id,
                body={"requests": [{"insertText": {"location": {"index": 1}, "text": text}}]},
            ).execute()
        logger.info("Created Google Doc '%s' (%s)", title, doc_id)
        return doc_id
    except Exception as exc:
        raise GDocsError(f"Failed to create document: {exc}") from exc


def _extract_text(doc: dict) -> str:
    """Extract plain text from a Google Docs API document object."""
    lines = []
    for element in doc.get("body", {}).get("content", []):
        paragraph = element.get("paragraph")
        if not paragraph:
            continue
        for elem in paragraph.get("elements", []):
            text_run = elem.get("textRun")
            if text_run:
                lines.append(text_run.get("content", ""))
    return "".join(lines)
