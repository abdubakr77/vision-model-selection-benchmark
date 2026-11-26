"""One wrapper per model, all exposing the same predict interface."""

from .yolo import predict_yolo

__all__ = ["predict_yolo"]
