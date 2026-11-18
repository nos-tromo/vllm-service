"""A user token signed the way Open WebUI signed one, for the tests that check it is ignored.

The router verified these until v0.12.0. A stray token that is validly signed
with a secret still set in the environment is what a revived verifier would
accept, so the tests send one instead of a malformed string. All data is
synthetic.
"""

import base64
import hashlib
import hmac
import json
import time

#: The secret a host may still set as ``AUDIT_OPENWEBUI_JWT_SECRET``; the tests sign their token with it.
LEFTOVER_SECRET = "synthetic-leftover-secret"
#: The e-mail the token names; it must never reach an event or a partition.
TOKEN_EMAIL = "alice@example.org"


def _b64(raw: bytes) -> str:
    """Encode bytes as unpadded base64url.

    Args:
        raw: The bytes.

    Returns:
        The encoded text.
    """
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def signed_token(secret: str) -> str:
    """Mint an HS256 user token, valid now, the way Open WebUI did.

    Args:
        secret: The signing secret.

    Returns:
        The token.
    """
    now = int(time.time())
    claims = {
        "sub": "0b7e6a2c-0000-4000-8000-0000000000aa",
        "email": TOKEN_EMAIL,
        "name": "Alice Example",
        "role": "user",
        "iss": "open-webui",
        "iat": now,
        "exp": now + 300,
    }
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = _b64(json.dumps(claims).encode())
    signature = hmac.new(secret.encode(), f"{header}.{body}".encode(), hashlib.sha256).digest()
    return f"{header}.{body}.{_b64(signature)}"
