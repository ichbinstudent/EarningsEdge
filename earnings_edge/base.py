"""
Abstract base class for all trading signal scanners.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseScanner(ABC):
    """
    Each scanner implements scan() and defines its cron schedule.
    Results follow a common schema for the Telegram bot to consume.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def scan(self) -> dict[str, Any]:
        """
        Run the scan and return results.

        Returns:
            Dict with keys:
            - 'success': bool
            - 'embed': dict with 'title', 'fields' list, 'timestamp'
              (each field: {'name': str, 'value': str, 'inline': bool})
            - 'error': str (if success=False)
            - 'timestamp': datetime
        """
        pass

