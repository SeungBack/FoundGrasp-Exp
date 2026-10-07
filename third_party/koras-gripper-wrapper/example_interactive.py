"""Interactive keyboard control for koras_gripper_wrapper.

Install once:  pip install -e .
Run:           python example_interactive.py [/dev/ttyACM0] [slave_id]

Type a command and press Enter (e.g. 'o', 'p 50', 'i').
"""

from __future__ import annotations

import sys

from koras_gripper_wrapper import KorasGripper

MENU = """
Commands:
  o            open
  c            close
  p <percent>  set position 0-100  (e.g. 'p 50')
  i            impedance on   (prompts type & stiffness)
  f            impedance off
  s            status
  e / d / x    enable / disable / stop
  h            help (this menu)
  q            quit
"""


def show_status(g: KorasGripper) -> None:
    s = g.read_status()
    if s is None:
        print("  (status read failed)")
        return
    print(f"  position={s.position:.1f}%  current={s.motor_cur}mA  "
          f"voltage={s.voltage}V  enabled={s.enabled}  fault={s.fault}")


def handle(g: KorasGripper, cmd: str, args: list[str]) -> None:
    if cmd == "o":
        print("opening ...", flush=True)
        g.open(wait=True)
        show_status(g)
    elif cmd == "c":
        print("closing ...", flush=True)
        g.close(wait=True)
        show_status(g)
    elif cmd == "p":
        if not args:
            print("  usage: p <percent 0-100>")
            return
        try:
            pct = float(args[0])
        except ValueError:
            print("  invalid number")
            return
        print(f"moving to {pct}% ...", flush=True)
        g.set_position(pct, wait=True)
        show_status(g)
    elif cmd == "i":
        gtype = input("  gripper type [1]: ").strip() or "1"
        stiff = input("  stiffness 1-10 [5]: ").strip() or "5"
        print("enabling impedance ...", flush=True)
        g.impedance_on(gripper_type=int(gtype), stiffness=int(stiff))
        show_status(g)
    elif cmd == "f":
        print("disabling impedance ...", flush=True)
        g.impedance_off()
        show_status(g)
    elif cmd == "s":
        show_status(g)
    elif cmd == "e":
        g.enable()
        print("  motor enabled")
    elif cmd == "d":
        g.disable()
        print("  motor disabled")
    elif cmd == "x":
        g.stop()
        print("  stopped")
    elif cmd == "h":
        print(MENU)
    else:
        print(f"  unknown command: {cmd!r} (type 'h' for help)")


def main() -> None:
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
    slave_id = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    print(f"Connecting to {port} (slave {slave_id}) ...", flush=True)
    g = KorasGripper(port, slave_id=slave_id)
    if not g.connect():  # auto enable + initialize (~5 s)
        print("  failed to connect")
        return
    print("  connected, motor enabled, gripper initialized")
    print(MENU)
    g.set_speed(100)
    print("  set gripper speed 100%")

    try:
        while True:
            try:
                raw = input("gripper> ").strip()
            except EOFError:
                break
            if not raw:
                continue
            cmd, *args = raw.split()
            if cmd.lower() == "q":
                break
            handle(g, cmd.lower(), args)
    except KeyboardInterrupt:
        pass
    finally:
        g.disconnect()
        print("\nDisconnected.")


if __name__ == "__main__":
    main()
