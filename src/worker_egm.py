import math
import socket
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from src import egm_pb2

EGM_HOST = "127.0.0.1"
EGM_PORT = 6510

V_MAX = 1.0
W_MAX = math.radians(180.0)
LEAD_MAX_M = 0.30
LEAD_MAX_RAD = math.radians(60.0)

FRAME_TOL_M = 0.005
FRAME_TOL_RAD = math.radians(2.0)
HOME_JOINT_TOL_DEG = 1.0

RX_TIMEOUT_S = 0.2
LINK_LOST_S = 1.0
DT_DEFAULT, DT_MIN, DT_MAX = 0.02, 0.001, 0.2


class _Session:
    def __init__(self):
        self.linked = False
        self.t_rx = 0.0
        self.t_tx = None
        self.ref_pos = None
        self.ref_rot = None
        self.q_prev = None
        self.armed = False
        self.checked = False
        self.seq = 0


class EgmWorker:
    def __init__(self, system, host=EGM_HOST, port=EGM_PORT):
        self.sys = system
        self.addr = (host, port)

    def _check_frame(self, st, fb_pos, fb_rot, joints):
        home = self.sys.shared_egm.read_home()
        if home is None:
            return
        st.checked = True
        q_home = np.array(home["q_deg"])
        if len(joints) < 6 or np.max(np.abs(np.array(joints[:6]) - q_home)) > HOME_JOINT_TOL_DEG:
            st.armed = True
            print("[EGM WARN] Robot not at home joints: frame check skipped")
            return
        w, x, y, z = home["quat_wxyz"]
        dp = np.linalg.norm(fb_pos - np.array(home["position"]))
        da = (fb_rot * R.from_quat([x, y, z, w]).inv()).magnitude()
        if dp <= FRAME_TOL_M and da <= FRAME_TOL_RAD:
            st.armed = True
            print(f"[EGM] Frame check OK (pos err {dp * 1000:.1f} mm, rot err {math.degrees(da):.2f} deg)")
        else:
            print(f"[EGM ERROR] Frame mismatch at home joints: pos err {dp * 1000:.1f} mm, rot err {math.degrees(da):.2f} deg")
            print(f"[EGM ERROR] feedback pos={np.round(fb_pos, 4).tolist()} expected={np.round(home['position'], 4).tolist()}")
            print("[EGM ERROR] Check tooldata TCP, work object and EGM frames. Holding position")

    @staticmethod
    def _advance(st, tgt, fb_pos, fb_rot, dt):
        tgt_pos = np.asarray(tgt["position"], dtype=float)
        w, x, y, z = tgt["quat_wxyz"]
        tgt_rot = R.from_quat([x, y, z, w])

        lead_p = np.linalg.norm(st.ref_pos - fb_pos)
        lead_a = (st.ref_rot * fb_rot.inv()).magnitude()

        d = tgt_pos - st.ref_pos
        dist = np.linalg.norm(d)
        max_d = V_MAX * dt
        cand_pos = tgt_pos.copy() if dist <= max_d else st.ref_pos + d * (max_d / dist)

        rv = (tgt_rot * st.ref_rot.inv()).as_rotvec()
        ang = np.linalg.norm(rv)
        max_a = W_MAX * dt
        if ang > max_a:
            rv = rv * (max_a / ang)
        cand_rot = R.from_rotvec(rv) * st.ref_rot

        if np.linalg.norm(cand_pos - fb_pos) <= max(LEAD_MAX_M, lead_p):
            st.ref_pos = cand_pos
        if (cand_rot * fb_rot.inv()).magnitude() <= max(LEAD_MAX_RAD, lead_a):
            st.ref_rot = cand_rot

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if hasattr(socket, "SIO_UDP_CONNRESET"):
            sock.ioctl(socket.SIO_UDP_CONNRESET, False)
        sock.bind(self.addr)
        sock.settimeout(RX_TIMEOUT_S)
        print(f"[EGM] Listening on {self.addr[0]}:{self.addr[1]}")

        st = _Session()
        try:
            while self.sys.is_running:
                try:
                    data, peer = sock.recvfrom(4096)
                except socket.timeout:
                    if st.linked and time.perf_counter() - st.t_rx > LINK_LOST_S:
                        print("[EGM] Link lost")
                        st = _Session()
                        self.sys.shared_egm.write_status({"linked": False})
                    continue
                except OSError:
                    continue

                t_in = time.perf_counter()
                robot = egm_pb2.EgmRobot()
                try:
                    robot.ParseFromString(data)
                except Exception:
                    continue
                fb = robot.feedBack
                if not (fb.HasField("cartesian") and fb.cartesian.HasField("pos") and fb.cartesian.HasField("orient")):
                    continue

                c = fb.cartesian
                fb_pos = np.array([c.pos.x, c.pos.y, c.pos.z]) / 1000.0
                fb_rot = R.from_quat([c.orient.u1, c.orient.u2, c.orient.u3, c.orient.u0])
                joints = list(fb.joints.joints)

                st.t_rx = t_in
                if not st.linked:
                    st.linked = True
                    print(f"[EGM] Link up from {peer[0]}:{peer[1]}")
                if st.ref_pos is None:
                    st.ref_pos, st.ref_rot = fb_pos.copy(), fb_rot
                if not st.checked:
                    self._check_frame(st, fb_pos, fb_rot, joints)

                dt = DT_DEFAULT if st.t_tx is None else min(max(t_in - st.t_tx, DT_MIN), DT_MAX)
                st.t_tx = t_in

                if not st.armed or not self.sys.teleop_active:
                    st.ref_pos, st.ref_rot = fb_pos.copy(), fb_rot
                else:
                    tgt = self.sys.shared_egm.read_target()
                    if tgt is not None:
                        self._advance(st, tgt, fb_pos, fb_rot, dt)

                q = st.ref_rot.as_quat()
                if st.q_prev is not None and float(np.dot(q, st.q_prev)) < 0.0:
                    q = -q
                st.q_prev = q

                st.seq += 1
                reply = egm_pb2.EgmSensor()
                reply.header.seqno = st.seq
                reply.header.tm = int(time.time() * 1000) & 0xFFFFFFFF
                reply.header.mtype = egm_pb2.EgmHeader.MSGTYPE_CORRECTION
                p = reply.planned.cartesian
                p.pos.x, p.pos.y, p.pos.z = (st.ref_pos * 1000.0).tolist()
                p.orient.u0, p.orient.u1, p.orient.u2, p.orient.u3 = q[3], q[0], q[1], q[2]
                sock.sendto(reply.SerializeToString(), peer)
                proc_ms = (time.perf_counter() - t_in) * 1000.0

                pb_state = getattr(self.sys, "shared_pb_state", None)
                if pb_state is not None:
                    fq = [c.orient.u0, c.orient.u1, c.orient.u2, c.orient.u3]
                    q_rad = np.radians(joints[:6]).tolist() if len(joints) >= 6 else [0.0] * 6
                    pb_state.write({"tcp_pos": fb_pos.tolist(), "tcp_quat_wxyz": fq, "q": q_rad}, time.time())

                self.sys.shared_egm.write_status({
                    "linked": True, "armed": st.armed,
                    "lead_mm": float(np.linalg.norm(st.ref_pos - fb_pos) * 1000.0),
                    "motors": int(robot.motorState.state), "mci": int(robot.mciState.state),
                    "rapid": int(robot.rapidExecState.state), "proc_ms": proc_ms,
                })
        finally:
            sock.close()
            print("[EGM] Closed")
