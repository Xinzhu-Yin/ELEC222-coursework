import sys, time, traceback
from dataclasses import dataclass
import serial
import numpy as np
from picamera2 import Picamera2
import classifier

@dataclass
class Config:
    serial_port: str = "/dev/ttyACM0"
    baudrate: int = 115200
    serial_timeout_s: float = 0.2
    width: int = 640
    height: int = 480
    max_line_len: int = 128

def parse_req(line: str):
    line = line.strip()
    if not line:
        return None
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        return None
    if parts[0].upper() != "REQ":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None

def send_line(ser: serial.Serial, s: str):
    ser.write((s + "\n").encode("utf-8", errors="ignore"))
    ser.flush()

class Camera:
    def __init__(self, cfg: Config):
        self.picam2 = Picamera2()
        conf = self.picam2.create_preview_configuration(
            main={"size": (cfg.width, cfg.height), "format": "RGB888"}
        )
        self.picam2.configure(conf)
        self.picam2.start()
        time.sleep(0.5)

    def capture(self) -> np.ndarray:
        return self.picam2.capture_array()

def main():
    cfg = Config()
    if len(sys.argv) >= 2:
        cfg.serial_port = sys.argv[1]

    cam = Camera(cfg)
    ser = serial.Serial(cfg.serial_port, cfg.baudrate, timeout=cfg.serial_timeout_s)
    time.sleep(0.3)
    ser.reset_input_buffer()

    last_seq = None

    while True:
        try:
            raw = ser.readline(cfg.max_line_len)
            if not raw:
                continue

            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            seq = parse_req(line)
            if seq is None:
                continue

            last_seq = seq

            # --- ACK placeholder (future) ---
            # send_line(ser, f"ACK,{seq}")

            img = cam.capture()
            class_id = classifier.predict(img)

            send_line(ser, f"RES,{seq},{int(class_id)}")

        except KeyboardInterrupt:
            break

        except Exception:
            traceback.print_exc()

            # --- ERR placeholder (future) ---
            # if last_seq is not None:
            #     send_line(ser, f"ERR,{last_seq},PI_EXCEPTION")
            # else:
            #     send_line(ser, "ERR,0,PI_EXCEPTION")

            time.sleep(0.1)
            continue

    try:
        ser.close()
    except Exception:
        pass
    return 0

if __name__ == "__main__":
    raise SystemExit(main())