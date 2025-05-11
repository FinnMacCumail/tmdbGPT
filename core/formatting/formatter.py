# core/formatting/formatter.py

from core.formatting.registry import RESPONSE_RENDERERS
# Ensure all registered renderers are loaded
from core.formatting.templates import *


class ResponseFormatter:
    @staticmethod
    def format_responses(state) -> list:
        """
        Dynamically dispatch response formatting based on response_format.
        Returns a list of formatted strings.
        """
        fmt = getattr(state, "response_format", "summary")
        renderer = RESPONSE_RENDERERS.get(fmt)

        if renderer:
            result = renderer(state)
            if isinstance(result, dict):
                return result.get("entries", [])
            elif isinstance(result, list):
                return result
            else:
                return [str(result)]

        return ["⚠️ No formatter available for response format: " + str(fmt)]
