"""Example usage of koras_gripper_wrapper.

Install once:  pip install -e .
Run:           python example_basic.py [/dev/ttyACM0] [slave_id]
"""

import sys

from koras_gripper_wrapper import KorasGripper


def show_status(g: KorasGripper) -> None:
    s = g.read_status()
    if s is None:
        print("  (status read failed)")
        return
    print(f"  position={s.position:.1f}%  current={s.motor_cur}mA  "
          f"voltage={s.voltage}V  enabled={s.enabled}  fault={s.fault}")


def basic_demo(g: KorasGripper) -> None:
    print("\n== Basic: open / close / position ==", flush=True)
    print("Opening ...", flush=True)
    g.open(wait=True)
    print("Closing ...", flush=True)
    g.close(wait=True)
    print("Moving to 50% ...", flush=True)
    g.set_position(50, wait=True)
    show_status(g)


def impedance_demo(g: KorasGripper) -> None:
    print("\n== Impedance control ==", flush=True)
    # impedance_on stays active for every following motion until impedance_off().
    print("Enabling impedance (type 1, stiffness 1) ...", flush=True)
    g.impedance_on(gripper_type=1, stiffness=1)
    print("Closing under impedance ...", flush=True)
    g.close(wait=True)
    print("Raising stiffness to 10 ...", flush=True)
    g.set_impedance(gripper_type=1, stiffness=10)
    print("Opening under impedance ...", flush=True)
    g.open(wait=True)
    print("Disabling impedance ...", flush=True)
    g.impedance_off()


def main() -> None:
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
    slave_id = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    print(f"Connecting to {port} (slave {slave_id}) ...", flush=True)
    g = KorasGripper(port, slave_id=slave_id)
    if not g.connect():  # auto enable + initialize (~5 s)
        print("  failed to connect")
        return
    print("  connected, motor enabled, gripper initialized")
    g.set_speed(100)
    print("  set gripper speed 100%")
    try:
        basic_demo(g)
        impedance_demo(g)
    finally:
        g.disconnect()
        print("\nDisconnected.")


if __name__ == "__main__":
    main()
