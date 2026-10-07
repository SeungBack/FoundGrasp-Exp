"""Minimal Python wrapper for the KORAS Robotics 2FPA-I-03-100 (KR-DATC) gripper.

Speaks Modbus RTU directly to the gripper over a serial port using pymodbus,
mirroring the official KR_GCS_user_interface behaviour.

Typical use
-----------
>>> with KorasGripper("/dev/ttyACM0") as g:   # connect() auto-enables + initializes
...     g.open(wait=True)
...     g.close(wait=True)
...     g.set_position(50, wait=True)         # 50 %
...     g.impedance_on(gripper_type=1, stiffness=5)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from pymodbus.client import ModbusSerialClient

from . import _const as C

log = logging.getLogger("koras_gripper_wrapper")

# Serial frame parameters (match modbus_comm.hpp: 8N1)
_DATA_BIT = 8
_STOP_BIT = 1
_PARITY = "N"


def _clamp(value: int, lo: int, hi: int, name: str) -> int:
    """Clamp ``value`` into ``[lo, hi]``, warning if it was out of range."""
    v = int(value)
    if v < lo or v > hi:
        log.warning("%s=%s out of range [%d, %d]; clamped", name, v, lo, hi)
        v = max(lo, min(hi, v))
    return v


def _clamp_signed(value: int, lo_abs: int, hi_abs: int, name: str) -> int:
    """Clamp the magnitude of ``value`` into ``[lo_abs, hi_abs]``, keeping sign."""
    v = int(value)
    mag = abs(v)
    if mag < lo_abs or mag > hi_abs:
        log.warning("%s=%s out of range (|v| in [%d, %d]); clamped", name, v, lo_abs, hi_abs)
        mag = max(lo_abs, min(hi_abs, mag))
    return mag if v >= 0 else -mag


def _s16(reg: int) -> int:
    """Interpret an unsigned 16-bit register as a signed int16."""
    return reg - 0x10000 if reg >= 0x8000 else reg


@dataclass
class GripperStatus:
    """Decoded status read from Holding Registers 10~17."""

    states: int       # raw status bitfield (register 10)
    motor_pos: int    # motor position, deg (int16)
    motor_cur: int    # motor current, mA (int16)
    motor_vel: int    # motor velocity, rpm (int16)
    finger_pos: int   # finger position, 0 (closed) ~ 1000 (open)
    voltage: int      # bus voltage, V

    @property
    def position(self) -> float:
        """Finger position in percent (0~100)."""
        return self.finger_pos / 10.0

    @property
    def enabled(self) -> bool:
        return bool(self.states & (1 << C.BIT_MOTOR_ENABLE))

    @property
    def initialized(self) -> bool:
        return bool(self.states & (1 << C.BIT_GRIPPER_INITIALIZE))

    @property
    def motor_pos_ctrl(self) -> bool:
        return bool(self.states & (1 << C.BIT_MOTOR_POS_CTRL))

    @property
    def motor_vel_ctrl(self) -> bool:
        return bool(self.states & (1 << C.BIT_MOTOR_VEL_CTRL))

    @property
    def motor_cur_ctrl(self) -> bool:
        return bool(self.states & (1 << C.BIT_MOTOR_CUR_CTRL))

    @property
    def grip_open(self) -> bool:
        return bool(self.states & (1 << C.BIT_GRIPPER_OPEN))

    @property
    def grip_close(self) -> bool:
        return bool(self.states & (1 << C.BIT_GRIPPER_CLOSE))

    @property
    def fault(self) -> bool:
        return bool(self.states & (1 << C.BIT_MOTOR_FAULT))


class KorasGripper:
    """Modbus RTU wrapper for a single KR-DATC gripper.

    Parameters
    ----------
    port:
        Serial device, e.g. ``"/dev/ttyACM0"`` or ``"COM3"``.
    slave_id:
        Modbus slave address of the gripper (factory default: 1).
    baudrate:
        Serial baud rate (default 115200, matching the official software).
    timeout:
        Per-request timeout in seconds.

    Control methods return ``bool`` (``True`` on success, ``False`` on a Modbus
    error). The high-level motion methods (:meth:`open`, :meth:`close`,
    :meth:`set_position`) accept ``wait=True`` to block until motion settles.
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        slave_id: int = 1,
        baudrate: int = 115200,
        timeout: float = 1.0,
    ) -> None:
        self._id = slave_id
        self._client = ModbusSerialClient(
            port,
            baudrate=baudrate,
            bytesize=_DATA_BIT,
            parity=_PARITY,
            stopbits=_STOP_BIT,
            timeout=timeout,
        )  # framer defaults to RTU for the serial client

    # ==================================================================
    # Connection
    # ==================================================================
    def connect(self, initialize: bool = True, init_timeout: float = 8.0) -> bool:
        """Open the serial port.

        If ``initialize`` is True (default), also enable the motor and run a
        gripper initialize, blocking until it finishes (or ``init_timeout``).
        Returns ``True`` once connected (initialize errors are logged, not fatal).
        """
        if not self._client.connect():
            log.error("failed to open serial port")
            return False
        if initialize:
            self.motor_enable()
            if self.grip_initialize():
                self._wait_initialize(init_timeout)
            else:
                log.error("gripper initialize failed during connect()")
        return True

    def disconnect(self) -> None:
        """Close the serial port."""
        self._client.close()

    @property
    def is_connected(self) -> bool:
        return self._client.connected

    @property
    def slave_id(self) -> int:
        return self._id

    def __enter__(self) -> "KorasGripper":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()

    # ==================================================================
    # High-level API (recommended)
    # ==================================================================
    def open(self, wait: bool = False, timeout: float = 10.0) -> bool:
        """Open the gripper. With ``wait=True``, block until motion settles."""
        if not self.grip_open():
            return False
        return self._wait_motion(timeout) if wait else True

    def close(self, wait: bool = False, timeout: float = 10.0) -> bool:
        """Close the gripper. With ``wait=True``, block until motion settles."""
        if not self.grip_close():
            return False
        return self._wait_motion(timeout) if wait else True

    def set_position(self, percent: float, wait: bool = False, timeout: float = 10.0) -> bool:
        """Move fingers to ``percent`` open (0 = closed, 100 = fully open).

        Resolution is 0.1 % (e.g. ``50.5``). With ``wait=True``, block until
        the motion settles.
        """
        raw = _clamp(round(percent * 10), C.FINGER_POS_MIN, C.FINGER_POS_MAX, "position(%)")
        if not self.set_finger_position(raw):
            return False
        return self._wait_motion(timeout) if wait else True

    def impedance_on(
        self,
        gripper_type: int,
        stiffness: int,
        wait: bool = True,
        init_timeout: float = 8.0,
    ) -> bool:
        """Enable impedance control and set its parameters.

        Runs the official sequence: impedance-on command -> gripper initialize
        -> set impedance parameters. ``gripper_type`` (1~20) selects a supported
        gripper, ``stiffness`` (1~10) sets grip softness (1 = softest).
        """
        ok = self._send(C.CMD_IMPEDANCE_ON)
        time.sleep(0.1)
        if self.grip_initialize() and wait:
            self._wait_initialize(init_timeout)
        return ok and self.set_impedance(gripper_type, stiffness)

    def impedance_off(self, wait: bool = True, init_timeout: float = 8.0) -> bool:
        """Disable impedance control (impedance-off command -> initialize)."""
        ok = self._send(C.CMD_IMPEDANCE_OFF)
        time.sleep(0.1)
        if self.grip_initialize() and wait:
            self._wait_initialize(init_timeout)
        return ok

    def set_impedance(self, gripper_type: int, stiffness: int) -> bool:
        """Update impedance parameters (gripper type 1~20, stiffness 1~10)."""
        t = _clamp(gripper_type, C.IMPEDANCE_TYPE_MIN, C.IMPEDANCE_TYPE_MAX, "gripper_type")
        s = _clamp(stiffness, C.STIFFNESS_MIN, C.STIFFNESS_MAX, "stiffness")
        return self._send(C.CMD_SET_IMPEDANCE_PARAMS, t, s)

    # Convenience aliases for the common motor commands
    def enable(self) -> bool:
        """Enable the motor (alias of :meth:`motor_enable`)."""
        return self.motor_enable()

    def disable(self) -> bool:
        """Disable the motor (alias of :meth:`motor_disable`)."""
        return self.motor_disable()

    def stop(self) -> bool:
        """Stop and hold at the current position (alias of :meth:`motor_stop`)."""
        return self.motor_stop()

    def initialize(self, wait: bool = True, timeout: float = 8.0) -> bool:
        """Run gripper initialize; with ``wait`` block until it finishes."""
        if not self.grip_initialize():
            return False
        return self._wait_initialize(timeout) if wait else True

    def set_force(self, percent: int) -> bool:
        """Set grip force as % of default torque (50~100); alias of set_motor_torque."""
        return self.set_motor_torque(percent)

    def set_speed(self, percent: int) -> bool:
        """Set motion speed as % of default speed (0~100); alias of set_motor_speed."""
        return self.set_motor_speed(percent)

    # ==================================================================
    # Wait helpers
    # ==================================================================
    def _wait_motion(self, timeout: float, tol: int = 3, stable_n: int = 3,
                     poll: float = 0.05) -> bool:
        """Block until the finger position stops changing (within ``tol``).

        Works uniformly for open / close / set_position, including the case of
        gripping an object (motion stops early). Returns ``False`` on timeout
        or a read error.
        """
        deadline = time.monotonic() + timeout
        time.sleep(0.1)  # let motion begin before sampling
        last = None
        stable = 0
        while time.monotonic() < deadline:
            s = self.read_status()
            if s is None:
                return False
            if last is not None and abs(s.finger_pos - last) <= tol:
                stable += 1
                if stable >= stable_n:
                    return True
            else:
                stable = 0
            last = s.finger_pos
            time.sleep(poll)
        log.warning("wait timed out after %.1fs", timeout)
        return False

    def _wait_initialize(self, timeout: float, poll: float = 0.05) -> bool:
        """Block until the gripper-initialize status bit goes busy then clears."""
        deadline = time.monotonic() + timeout
        time.sleep(0.1)
        saw_busy = False
        while time.monotonic() < deadline:
            s = self.read_status()
            if s is not None:
                if s.initialized:
                    saw_busy = True
                elif saw_busy:
                    return True
            time.sleep(poll)
        if not saw_busy:
            return True  # init was too quick to observe; assume done
        log.warning("initialize wait timed out after %.1fs", timeout)
        return False

    # ==================================================================
    # Transport core (mirrors SEND_CMD / SEND_CMD_VECTOR / recvData)
    # ==================================================================
    def _send(self, cmd: int, *values: int) -> bool:
        """Write a command to Holding Register 0.

        Single value -> write_register (func 0x06); multiple -> write_registers
        (func 0x10), exactly like the official firmware. Negative values are
        sent as two's-complement 16-bit.
        """
        if not self._client.connected:
            log.error("not connected")
            return False
        try:
            if values:
                payload = [cmd & 0xFFFF] + [v & 0xFFFF for v in values]
                rr = self._client.write_registers(C.CMD_ADDR, payload, device_id=self._id)
            else:
                rr = self._client.write_register(C.CMD_ADDR, cmd & 0xFFFF, device_id=self._id)
        except Exception as e:  # pragma: no cover - hardware/IO errors
            log.error("modbus write failed: %s", e)
            return False
        if rr.isError():
            log.error("modbus error on cmd %d: %s", cmd, rr)
            return False
        return True

    # ==================================================================
    # Low-level commands (building blocks, 1:1 with the official DatcCtrl)
    # ==================================================================
    def motor_enable(self) -> bool:
        """Enable the motor (holds current angle via position control)."""
        return self._send(C.CMD_MOTOR_ENABLE)

    def motor_stop(self) -> bool:
        """Stop and hold the motor at the current position."""
        return self._send(C.CMD_MOTOR_STOP)

    def motor_disable(self) -> bool:
        """Cut power to the motor."""
        return self._send(C.CMD_MOTOR_DISABLE)

    def motor_position_control(self, pos_deg: int, duration: int = 500) -> bool:
        """Move the motor to ``pos_deg`` (absolute degrees) over ``duration`` ms."""
        duration = _clamp(duration, C.DURATION_MIN, C.DURATION_MAX, "duration")
        return self._send(C.CMD_MOTOR_POSITION_CONTROL, pos_deg, duration)

    def motor_velocity_control(self, vel_rpm: int) -> bool:
        """Velocity control. ``vel_rpm`` magnitude clamped to [100, 900]."""
        vel = _clamp_signed(vel_rpm, C.VEL_MIN, C.VEL_MAX, "velocity")
        return self._send(C.CMD_MOTOR_VELOCITY_CONTROL, vel, 500)

    def motor_current_control(self, cur_ma: int) -> bool:
        """Current control. ``cur_ma`` magnitude clamped to <= 1200 mA."""
        cur = _clamp_signed(cur_ma, 0, C.CUR_MAX, "current")
        return self._send(C.CMD_MOTOR_CURRENT_CONTROL, cur, 500)

    def grip_initialize(self) -> bool:
        """Detect gripper type and stroke. Required once after power-on."""
        return self._send(C.CMD_GRIPPER_INITIALIZE)

    def grip_open(self) -> bool:
        """Open the gripper (no wait); see :meth:`open` for the wait option."""
        return self._send(C.CMD_GRIPPER_OPEN)

    def grip_close(self) -> bool:
        """Close the gripper (no wait); see :meth:`close` for the wait option."""
        return self._send(C.CMD_GRIPPER_CLOSE)

    def set_finger_position(self, finger_pos: int) -> bool:
        """Set raw finger position, 0 (closed) ~ 1000 (open), in 0.1% steps."""
        finger_pos = _clamp(finger_pos, C.FINGER_POS_MIN, C.FINGER_POS_MAX, "finger_pos")
        return self._send(C.CMD_SET_FINGER_POSITION, finger_pos)

    def vacuum_on(self) -> bool:
        """Start the vacuum-gripper pump."""
        return self._send(C.CMD_VACUUM_GRIPPER_ON)

    def vacuum_off(self) -> bool:
        """Stop the vacuum-gripper pump."""
        return self._send(C.CMD_VACUUM_GRIPPER_OFF)

    def set_motor_torque(self, torque_ratio: int) -> bool:
        """Set grip force as % of default torque, clamped to [50, 100]."""
        r = _clamp(torque_ratio, C.TORQUE_RATIO_MIN, C.TORQUE_RATIO_MAX, "torque_ratio")
        return self._send(C.CMD_SET_MOTOR_TORQUE, r)

    def set_motor_speed(self, speed_ratio: int) -> bool:
        """Set motion speed as % of default speed, clamped to [0, 100]."""
        r = _clamp(speed_ratio, C.SPEED_RATIO_MIN, C.SPEED_RATIO_MAX, "speed_ratio")
        return self._send(C.CMD_SET_MOTOR_SPEED, r)

    def set_impedance_params(self, gripper_type: int, stiffness_level: int) -> bool:
        """Low-level: set impedance type (1~20) and stiffness (1~10)."""
        return self.set_impedance(gripper_type, stiffness_level)

    def set_modbus_address(self, slave_addr: int) -> bool:
        """Change the gripper's stored slave address.

        Takes effect only after the gripper is power-cycled. The active
        connection keeps using the current ``slave_id`` until then.
        """
        if not (C.SLAVE_ADDR_MIN <= slave_addr <= C.SLAVE_ADDR_MAX):
            log.error("set_modbus_address: addr %s out of range [%d, %d]",
                      slave_addr, C.SLAVE_ADDR_MIN, C.SLAVE_ADDR_MAX)
            return False
        return self._send(C.CMD_CHANGE_MODBUS_ADDRESS, slave_addr)

    # ==================================================================
    # Status (synchronous read)
    # ==================================================================
    def read_status(self) -> GripperStatus | None:
        """Read and decode status registers. Returns ``None`` on a read error."""
        if not self._client.connected:
            log.error("not connected")
            return None
        try:
            rr = self._client.read_holding_registers(
                C.STATUS_ADDR, count=C.STATUS_LEN, device_id=self._id
            )
        except Exception as e:  # pragma: no cover - hardware/IO errors
            log.error("modbus read failed: %s", e)
            return None
        if rr.isError():
            log.error("modbus error on status read: %s", rr)
            return None
        r = rr.registers
        return GripperStatus(
            states=r[0],
            motor_pos=_s16(r[1]),
            motor_cur=_s16(r[2]),
            motor_vel=_s16(r[3]),
            finger_pos=r[4],
            voltage=r[7],
        )
