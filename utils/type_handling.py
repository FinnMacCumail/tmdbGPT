from datetime import datetime
import re
from typing import Any

TYPE_CONVERTERS = {
    "integer": lambda v: int(float(v)) if isinstance(v, str) else int(v),
    "number": float,
    "boolean": lambda v: str(v).lower() in ["true", "1", "yes"],
    "date": lambda v: datetime.strptime(v, "%Y-%m-%d").date(),
    "year": lambda v: int(re.sub(r"\D", "", str(v))[:4])
}

def convert_value(value: Any, target_type: str) -> Any:
    try:
        return TYPE_CONVERTERS[target_type](value)
    except KeyError:
        print(f"🚨 Unknown type {target_type}, using string conversion")
        return str(value)
    except Exception as e:
        print(f"⚠️ Soft fail: Couldn't convert {value} to {target_type}")
        return value  # Preserve original for error recovery