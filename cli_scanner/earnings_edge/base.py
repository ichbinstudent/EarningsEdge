"""
Abstract base class for all trading signal scanners.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseScanner(ABC):
    """
    Each scanner implements scan() and defines its cron schedule.
    Results follow a common schema for the Telegram bot to consume.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def scan(self) -> Dict[str, Any]:
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

    @property
    @abstractmethod
    def schedule(self) -> str:
        """Cron schedule string, e.g. '30 21 * * mon-fri'.

        NOTE: APScheduler numbers weekdays Monday=0 — use names (mon-fri),
        never '1-5' (that means Tue-Sat and skips Monday)."""
        pass

    def get_schedule_description(self) -> str:
        return f"Scanner '{self.name}' runs on schedule: {self.schedule}"
