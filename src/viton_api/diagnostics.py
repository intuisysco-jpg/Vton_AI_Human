"""Small, bounded diagnostic messages that never expose service credentials."""

import re


def safe_reason(value, settings) -> str:
    value = str(value)
    for attribute in ("database_url", "rabbitmq_url"):
        secret = getattr(settings, attribute, "")
        if secret:
            value = value.replace(secret, "[redacted connection URL]")
    value = re.sub(r"([a-zA-Z][a-zA-Z0-9+.-]*://)[^\s/@]+@", r"\1[redacted]@", value)
    value = re.sub(r"hf_[A-Za-z0-9]{8,}", "[redacted token]", value)
    value = re.sub(
        r"(?i)((?:token|password|secret|api_key|cap)=)[^\s&]+", r"\1[redacted]", value
    )
    return " ".join(value.split())[:300]
