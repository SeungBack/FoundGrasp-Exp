"""DATC / KR-GCS Modbus command codes and value ranges.

Mirrors the official KR_GCS_user_interface firmware constants
(see include/datc_ctrl.hpp). All gripper control goes through
Holding Register 0 (command) and status comes from Holding
Registers 10~17.
"""

# --- Holding register addresses ---
CMD_ADDR = 0      # write: [command, value_1, value_2, ...]
STATUS_ADDR = 10  # read: 8 registers (status .. voltage)
STATUS_LEN = 8

# --- Command codes (Holding Register 0) ---
CMD_MOTOR_ENABLE = 1
CMD_MOTOR_STOP = 2
CMD_MOTOR_DISABLE = 4
CMD_MOTOR_POSITION_CONTROL = 5
CMD_MOTOR_VELOCITY_CONTROL = 6
CMD_MOTOR_CURRENT_CONTROL = 7
CMD_CHANGE_MODBUS_ADDRESS = 50
CMD_GRIPPER_INITIALIZE = 101
CMD_GRIPPER_OPEN = 102
CMD_GRIPPER_CLOSE = 103
CMD_SET_FINGER_POSITION = 104
CMD_VACUUM_GRIPPER_ON = 106
CMD_VACUUM_GRIPPER_OFF = 107
CMD_IMPEDANCE_ON = 108
CMD_IMPEDANCE_OFF = 109
CMD_SET_IMPEDANCE_PARAMS = 110
CMD_SET_MOTOR_TORQUE = 212
CMD_SET_MOTOR_SPEED = 213

# --- Value ranges (from datc_ctrl.hpp) ---
FINGER_POS_MIN, FINGER_POS_MAX = 0, 1000     # 0: closed, 1000: open (0.1% steps)
DURATION_MIN, DURATION_MAX = 10, 10000       # ms
TORQUE_RATIO_MIN, TORQUE_RATIO_MAX = 50, 100 # %
SPEED_RATIO_MIN, SPEED_RATIO_MAX = 0, 100    # %
VEL_MIN, VEL_MAX = 100, 900                  # rpm (abs)
CUR_MAX = 1200                               # mA (abs)
IMPEDANCE_TYPE_MIN, IMPEDANCE_TYPE_MAX = 1, 20
STIFFNESS_MIN, STIFFNESS_MAX = 1, 10
SLAVE_ADDR_MIN, SLAVE_ADDR_MAX = 1, 99       # setModbusAddr: 1 <= addr < 100

# --- Status bit positions (Holding Register 10) ---
BIT_MOTOR_ENABLE = 0
BIT_GRIPPER_INITIALIZE = 1
BIT_MOTOR_POS_CTRL = 2
BIT_MOTOR_VEL_CTRL = 3
BIT_MOTOR_CUR_CTRL = 4
BIT_GRIPPER_OPEN = 5
BIT_GRIPPER_CLOSE = 6
BIT_MOTOR_FAULT = 9
