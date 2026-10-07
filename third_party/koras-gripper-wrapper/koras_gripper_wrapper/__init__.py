"""koras_gripper_wrapper - minimal Modbus RTU wrapper for the KORAS 2FPA-I-03-100 gripper."""

from .gripper import GripperStatus, KorasGripper

__all__ = ["KorasGripper", "GripperStatus"]
__version__ = "0.1.0"
