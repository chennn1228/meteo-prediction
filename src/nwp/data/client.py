"""No-overwrite JSON acquisition client with retry and receipt validation."""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from nwp.core.hashing import file_sha256
from nwp.core.provenance import read_receipt, write_receipt
from nwp.core.schema import ContractError


DEFAULT_RETRIES = 4
DEFAULT_TIMEOUT = 60
USER_AGENT = "nwp-postprocessing/2.1"


class DataFetchError(ContractError):
    """A remote request or immutable cache fails its acquisition contract."""


@dataclass(frozen=True)
class FetchOutcome:
    payload: Mapping[str, Any]
    data_path: Path
    receipt_path: Path
    sha256: str
    reused: bool


def build_url(endpoint: str, params: Mapping[str, Any]) -> str:
    if not endpoint.startswith("https://"):
        raise DataFetchError("data endpoints must use HTTPS")
    return endpoint + "?" + urllib.parse.urlencode(dict(params))


def _payload(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataFetchError(f"raw JSON is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise DataFetchError("raw JSON payload must be an object")
    return value


def _reuse(
    data_path: Path,
    receipt_path: Path,
    *,
    params: Mapping[str, Any],
    validator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> FetchOutcome:
    if data_path.exists() != receipt_path.exists():
        raise DataFetchError("raw data and receipt must either both exist or both be absent")
    payload = _payload(data_path)
    validator(payload)
    receipt = read_receipt(receipt_path)
    digest = file_sha256(data_path)
    if receipt.get("request_parameters") != dict(params):
        raise DataFetchError("cached raw request parameters differ from the requested contract")
    if receipt.get("data_sha256") != digest or digest not in receipt["output_hashes"].values():
        raise DataFetchError("cached raw content hash differs from its receipt")
    return FetchOutcome(payload, data_path, receipt_path, digest, True)


def fetch_json(
    endpoint: str,
    params: Mapping[str, Any],
    *,
    data_path: Path,
    receipt_path: Path,
    validator: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    receipt_builder: Callable[[Mapping[str, Any], Mapping[str, Any], str], Mapping[str, Any]],
    retries: int = DEFAULT_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    opener: Callable[..., Any] = urllib.request.urlopen,
    sleeper: Callable[[float], None] = time.sleep,
) -> FetchOutcome:
    """Fetch one immutable raw JSON object or validate and reuse its sidecar pair."""
    if retries < 1 or timeout < 1:
        raise DataFetchError("retries and timeout must be positive")
    if data_path.exists() or receipt_path.exists():
        return _reuse(data_path, receipt_path, params=params, validator=validator)

    request = urllib.request.Request(build_url(endpoint, params), headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    raw: str | None = None
    payload: Mapping[str, Any] | None = None
    audit: Mapping[str, Any] | None = None
    for attempt in range(1, retries + 1):
        try:
            with opener(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            parsed = json.loads(raw)
            if not isinstance(parsed, dict) or parsed.get("error"):
                raise DataFetchError(f"API returned an error payload: {parsed.get('reason') if isinstance(parsed, dict) else 'non-object'}")
            audit = validator(parsed)
            payload = parsed
            break
        except ContractError:
            raise
        except Exception as exc:  # transient HTTP/JSON errors are retried
            last_error = exc
            if attempt < retries:
                sleeper(float(2**attempt))
    if raw is None or payload is None or audit is None:
        raise DataFetchError(f"request retries exhausted: {last_error}")

    data_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = data_path.with_name(f".{data_path.name}.tmp")
    if temporary.exists():
        raise DataFetchError(f"stale raw temporary file exists: {temporary}")
    try:
        temporary.write_text(raw, encoding="utf-8")
        digest = file_sha256(temporary)
        temporary.replace(data_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    receipt = receipt_builder(payload, audit, digest)
    write_receipt(receipt_path, receipt)
    return FetchOutcome(payload, data_path, receipt_path, digest, False)
