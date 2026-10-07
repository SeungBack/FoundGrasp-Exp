# koras_gripper_wrapper

Minimal, easy-to-use Python wrapper for the **KORAS Robotics 2FPA-I-03-100 (KR-DATC)** gripper.

It talks **Modbus RTU** directly to the gripper over a serial port via
[pymodbus](https://github.com/pymodbus-dev/pymodbus), reproducing the exact
command frames of the official `KR_GCS_user_interface` software.

## Install

```bash
pip install -e .        # installs koras_gripper_wrapper + pymodbus + pyserial
```

Grant access to the serial port (once per boot):

```bash
sudo chmod 666 /dev/ttyACM0
```

## Quick start

```python
from koras_gripper_wrapper import KorasGripper

# connect() auto-enables the motor and initializes the gripper (~5 s).
with KorasGripper("/dev/ttyACM0", slave_id=1) as g:
    g.open(wait=True)              # fully open, block until done
    g.close(wait=True)            # fully close
    g.set_position(50, wait=True)  # go to 50 % open
    print(g.read_status().position)  # -> 50.0
```

- Position is in **percent (0~100)**, 0.1 % resolution.
- `wait=True` blocks until the motion settles (returns early when an object is
  gripped); omit it (default) to return immediately.
- Every control method returns `True` on success / `False` on a Modbus error.
  `read_status()` returns a `GripperStatus`, or `None` on a read error.

Impedance control (stays active until `impedance_off()`):

```python
g.impedance_on(gripper_type=1, stiffness=5)   # on -> initialize -> set params
g.close(wait=True)                            # all motion now uses impedance
g.set_impedance(gripper_type=1, stiffness=8)  # change stiffness on the fly
g.impedance_off()
```

Runnable demos: [example_basic.py](example_basic.py) (scripted sequence) and
[example_interactive.py](example_interactive.py) (keyboard control) &mdash; e.g.
`python example_basic.py /dev/ttyACM0 1`.

## API

### High-level (recommended)
- `connect(initialize=True)` &mdash; open the port; auto enable + initialize
- `open(wait=False)`, `close(wait=False)` &mdash; `wait=True` blocks until settled
- `set_position(percent, wait=False)` &mdash; 0~100 %, 0.1 % resolution
- `impedance_on(gripper_type, stiffness)` &mdash; on → initialize → set params
- `impedance_off()`
- `set_impedance(gripper_type, stiffness)` &mdash; type (1~20) / stiffness (1~10)
- `enable()`, `disable()`, `stop()`, `initialize(wait=True)`
- `set_force(50..100)` &mdash; grip force, % of default torque
- `set_speed(0..100)` &mdash; motion speed, % of default
- `read_status()` &rarr; `GripperStatus`

### `GripperStatus`
`GripperStatus(states, motor_pos, motor_cur, motor_vel, finger_pos, voltage)`:
- `.position` &mdash; finger position in percent (0~100)
- bit flags: `.enabled .initialized .motor_pos_ctrl .motor_vel_ctrl
  .motor_cur_ctrl .grip_open .grip_close .fault`

### Low-level building blocks
1:1 with the official `DatcCtrl`, no waiting:
- `motor_enable/stop/disable()`, `motor_position_control(deg, duration=500)`,
  `motor_velocity_control(rpm)`, `motor_current_control(mA)`
- `grip_initialize/open/close()`, `set_finger_position(0..1000)`
- `set_motor_torque(50..100)`, `set_motor_speed(0..100)`
- `vacuum_on/off()`, `set_impedance_params(type, stiffness)`
- `set_modbus_address(addr)` &mdash; persisted, takes effect after a power-cycle

## Communication

Modbus RTU over RS-485. Defaults match the official software:

| Param     | Value                       |
| --------- | --------------------------- |
| Baud rate | 115200                      |
| Frame     | 8 data, 1 stop, parity None |
| Slave ID  | 1 (default)                 |

- **Commands** are written to Holding Register `0` as `[command, value_1, value_2]`.
- **Status** is read from Holding Registers `10`~`17`.

## Notes
- `connect()` auto enable + initialize takes ~5 s (the gripper sweeps its full
  stroke). Use `connect(initialize=False)` to skip it.
- Out-of-range arguments are clamped with a `logging.warning`. Silence with
  `logging.getLogger("koras_gripper_wrapper").setLevel(logging.ERROR)`.
- Most USB&ndash;RS-485 adapters auto-toggle direction, so no extra RS-485 config
  is needed. If writes fail on a raw UART, enable pyserial RS-485 mode on the port.

The official manual is kept as [MANUAL.md](MANUAL.md) for reference. The source
PDF and the reference UI sources (`KR_GCS_user_interface/`) are git-ignored.
