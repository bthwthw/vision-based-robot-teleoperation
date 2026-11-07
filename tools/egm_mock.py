import argparse
import math
import os
import socket
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import egm_pb2

HOME_POS = [0.451, 0.0, 0.6461]
HOME_QUAT_WXYZ = [0.0, 0.70711, 0.70711, 0.0]
HOME_JOINTS_DEG = [0, 0, 0, 0, 90, 90]


class MockController:
    def __init__(self, host="127.0.0.1", port=6510, rate_hz=25.0, pos=HOME_POS, quat_wxyz=HOME_QUAT_WXYZ,
                 joints_deg=HOME_JOINTS_DEG, lag_s=0.1, vmax=1.0, frozen=False):
        self.addr = (host, port)
        self.period = 1.0 / rate_hz
        self.pos = np.array(pos, dtype=float)
        w, x, y, z = quat_wxyz
        self.rot = R.from_quat([x, y, z, w])
        self.joints_deg = list(joints_deg)
        self.lag_s = lag_s
        self.vmax = vmax
        self.frozen = frozen
        self.replies = []
        self.sent = 0
        self.latency_ms = []

    def _build(self, seq):
        m = egm_pb2.EgmRobot()
        m.header.seqno = seq
        m.header.tm = int(time.time() * 1000) & 0xFFFFFFFF
        m.header.mtype = egm_pb2.EgmHeader.MSGTYPE_DATA
        c = m.feedBack.cartesian
        c.pos.x, c.pos.y, c.pos.z = (self.pos * 1000.0).tolist()
        x, y, z, w = self.rot.as_quat()
        c.orient.u0, c.orient.u1, c.orient.u2, c.orient.u3 = w, x, y, z
        m.feedBack.joints.joints.extend(self.joints_deg)
        m.motorState.state = egm_pb2.EgmMotorState.MOTORS_ON
        m.mciState.state = egm_pb2.EgmMCIState.MCI_RUNNING
        m.rapidExecState.state = egm_pb2.EgmRapidCtrlExecState.RAPID_RUNNING
        return m.SerializeToString()

    def _apply(self, reply):
        p = reply.planned.cartesian
        ref_pos = np.array([p.pos.x, p.pos.y, p.pos.z]) / 1000.0
        ref_rot = R.from_quat([p.orient.u1, p.orient.u2, p.orient.u3, p.orient.u0])
        self.replies.append((ref_pos, ref_rot))
        if self.frozen:
            return
        a = 1.0 - math.exp(-self.period / self.lag_s)
        d = (ref_pos - self.pos) * a
        n = np.linalg.norm(d)
        if n > self.vmax * self.period:
            d *= self.vmax * self.period / n
        self.pos = self.pos + d
        self.rot = R.from_rotvec((ref_rot * self.rot.inv()).as_rotvec() * a) * self.rot

    def run(self, duration_s=None, stop=None, verbose=False):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 0))
        t_end = None if duration_s is None else time.perf_counter() + duration_s
        t_next = time.perf_counter()
        t_print = t_next
        seq = 0
        try:
            while (t_end is None or time.perf_counter() < t_end) and not (stop is not None and stop.is_set()):
                seq += 1
                t_send = time.perf_counter()
                sock.sendto(self._build(seq), self.addr)
                self.sent += 1
                sock.settimeout(self.period * 0.9)
                try:
                    data, _ = sock.recvfrom(4096)
                    self.latency_ms.append((time.perf_counter() - t_send) * 1000.0)
                    r = egm_pb2.EgmSensor()
                    r.ParseFromString(data)
                    self._apply(r)
                except (socket.timeout, ConnectionResetError):
                    pass
                t_next += self.period
                time.sleep(max(0.0, t_next - time.perf_counter()))
                if verbose and time.perf_counter() - t_print > 1.0:
                    t_print = time.perf_counter()
                    print(f"[MOCK] pos(mm)={np.round(self.pos * 1000, 1).tolist()} sent={self.sent} replies={len(self.replies)}")
        finally:
            sock.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fake ABB controller for EGM Pose mode")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6510)
    ap.add_argument("--rate", type=float, default=25.0)
    ap.add_argument("--pos", type=float, nargs=3, default=HOME_POS, help="start TCP pos (m)")
    ap.add_argument("--quat", type=float, nargs=4, default=HOME_QUAT_WXYZ, help="start TCP quat w x y z")
    ap.add_argument("--joints", type=float, nargs=6, default=HOME_JOINTS_DEG, help="reported joints (deg)")
    ap.add_argument("--lag", type=float, default=0.1)
    ap.add_argument("--vmax", type=float, default=1.0)
    a = ap.parse_args()
    mock = MockController(a.host, a.port, a.rate, a.pos, a.quat, a.joints, a.lag, a.vmax)
    print(f"[MOCK] Sending to {a.host}:{a.port} at {a.rate} Hz, start pos={a.pos}")
    try:
        mock.run(verbose=True)
    except KeyboardInterrupt:
        pass
