"""
Validator authentication for the TuneForge API.

Dual-mode authentication:
1. Hotkey signature (production): Validators sign each request with their
   Bittensor hotkey. The server verifies the signature and checks the
   hotkey is a registered validator on the metagraph.
2. Bearer token (dev/fallback): Simple shared-secret token for local
   development or when the metagraph is unavailable.
"""

import hashlib
import secrets
import time

from fastapi import HTTPException, Request, status
from loguru import logger

try:
    from bittensor_wallet import Keypair
except ImportError:
    Keypair = None  # type: ignore[assignment,misc]

# Redis key prefix for nonce deduplication
_NONCE_KEY_PREFIX = "validator_nonce:"


def _get_body_hash(body: bytes) -> str:
    """SHA-256 hex digest of the request body, or 'empty' if no body."""
    if not body:
        return "empty"
    return hashlib.sha256(body).hexdigest()


async def _check_nonce_reuse(request: Request, hotkey: str, nonce_str: str) -> None:
    """Check if this nonce has already been used. Raises 401 if replay detected."""
    redis_mgr = getattr(request.app.state, "redis", None)
    if redis_mgr is None:
        # No Redis available — skip dedup (freshness check still applies)
        logger.warning("Redis unavailable — nonce replay protection disabled")
        return

    key = f"{_NONCE_KEY_PREFIX}{hotkey}:{nonce_str}"
    # _client is the internal redis.asyncio.Redis instance
    was_set = await redis_mgr._client.set(key, "1", nx=True, ex=300)
    if not was_set:
        logger.warning("Replay detected: nonce {} reused for hotkey {}", nonce_str, hotkey[:16])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nonce already used (replay detected).",
        )


async def _verify_hotkey_signature(request: Request) -> str:
    """Verify a request signed with a Bittensor hotkey.

    Returns the validator's ss58 hotkey address on success.

    Order of checks:
    1. Parse headers
    2. Nonce freshness (time window)
    3. Metagraph registration + validator_permit
    4. Cryptographic signature verification
    5. Nonce deduplication via Redis (AFTER sig verify to prevent poisoning)
    """
    from tuneforge.settings import get_settings

    hotkey = request.headers.get("X-Validator-Hotkey", "")
    nonce_str = request.headers.get("X-Validator-Nonce", "")
    signature = request.headers.get("X-Validator-Signature", "")

    if not hotkey or not nonce_str or not signature:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing hotkey authentication headers.",
        )

    # Replay protection: check nonce freshness
    try:
        nonce = int(nonce_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid nonce format.",
        )

    settings = get_settings()
    now_ns = time.time_ns()
    if abs(now_ns - nonce) > settings.validator_nonce_tolerance_ns:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nonce expired or too far in the future.",
        )

    # Check metagraph registration
    from tuneforge.api.server import app_state
    metagraph = app_state.metagraph
    if metagraph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Metagraph not available — cannot verify hotkey.",
        )

    hotkeys = list(metagraph.hotkeys)
    if hotkey not in hotkeys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Hotkey not registered on the metagraph.",
        )

    idx = hotkeys.index(hotkey)
    validator_permits = list(metagraph.validator_permit)
    if idx >= len(validator_permits) or not validator_permits[idx]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Hotkey does not have validator permit.",
        )

    # Verify signature
    if Keypair is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="bittensor_wallet not installed — cannot verify signatures.",
        )

    body = await request.body()
    body_hash = _get_body_hash(body)
    # Use decoded path WITHOUT query string — must match client's request.url.path
    path = request.url.path
    method = request.method.upper()
    message = f"{nonce_str}.{hotkey}.{method}.{path}.{body_hash}"

    try:
        sig_bytes = bytes.fromhex(signature.removeprefix("0x"))
        kp = Keypair(ss58_address=hotkey)
        if not kp.verify(message, sig_bytes):
            raise ValueError("Signature verification returned False")
    except Exception as exc:
        logger.warning("Hotkey signature verification failed for {}: {}", hotkey[:16], exc)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid hotkey signature.",
        )

    # Replay protection: store nonce in Redis AFTER signature is verified
    # This prevents nonce poisoning (attacker burning nonces with fake signatures)
    await _check_nonce_reuse(request, hotkey, nonce_str)

    return hotkey


async def _verify_bearer_token(request: Request) -> str:
    """Verify a Bearer token from the Authorization header.

    Returns 'bearer-token' on success.
    """
    from tuneforge.settings import get_settings

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header.",
        )

    token = auth_header[7:]  # Strip "Bearer "
    expected = get_settings().validator_service_token
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Validator API not configured (no service token set).",
        )

    if not secrets.compare_digest(token, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid validator token.",
        )

    logger.warning("Validator authenticated via Bearer token (dev/fallback mode)")
    return "bearer-token"


async def require_validator_auth(request: Request) -> str:
    """FastAPI dependency: authenticate a validator request.

    Supports two modes:
    1. Hotkey signature: X-Validator-Hotkey + X-Validator-Nonce +
       X-Validator-Signature headers present → verify cryptographic signature
       against the Bittensor metagraph.
    2. Bearer token: Authorization: Bearer <token> header → verify shared secret.

    Returns the validator's ss58 hotkey (mode 1) or 'bearer-token' (mode 2).
    """
    if request.headers.get("X-Validator-Hotkey"):
        return await _verify_hotkey_signature(request)

    if request.headers.get("Authorization", "").startswith("Bearer "):
        return await _verify_bearer_token(request)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No validator authentication provided. "
               "Use hotkey signing headers or Bearer token.",
    )
