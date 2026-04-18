"""Project-wide exception hierarchy."""


class GameAgentError(Exception):
    """Base exception for all game agent errors."""


class DeviceError(GameAgentError):
    """Base for device-related errors."""


class DeviceNotConnectedError(DeviceError):
    """Raised when the device is not reachable."""


class PocoNodeNotFoundError(DeviceError):
    """Raised when a Poco node path cannot be resolved."""


class ToolExecutionError(GameAgentError):
    """Raised when a tool fails to execute."""


class PerceptionError(GameAgentError):
    """Raised on perception capture failures."""


class GraphError(GameAgentError):
    """Base for graph engine errors."""


class NavigationError(GraphError):
    """Raised when graph navigation fails (no path, unreachable target)."""


class GeminiError(GameAgentError):
    """Base for Gemini API errors."""


class GeminiRateLimitError(GeminiError):
    """Raised when Gemini API rate limit is hit."""


class ConfigError(GameAgentError):
    """Raised on configuration loading/validation failures."""
