"""
Custom exception types for incwear.
"""


class IncwearError(Exception):
    """Base exception for all incwear-specific erros."""


class InvalidSensorFormatError(IncwearError):
    """Raised when an input file format or structure is invalid."""


class ThresholdComputationError(IncwearError):
    """Raised when movement thresholds cannot be computed reliably."""


class CalibrationError(IncwearError):
    """Raised when calibration cannot be completed or produces invalid output."""


class TimingError(IncwearError):
    """Raised when timestamp data are invalid or inconsistent."""
