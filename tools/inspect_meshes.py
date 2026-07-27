"""
tools/inspect_stl.py

Kiem tra nhanh 1 file .STL co phai mesh that hay khong (khong phai file rong/
placeholder/loi), bang cach tu doc header STL nhi phan va tinh bounding box.
KHONG can cai them thu vien ngoai (khong dung numpy-stl/trimesh).

Cach dung:
    python tools/inspect_stl.py "duong/dan/toi/file.STL"
"""

import struct
import sys
from pathlib import Path


def inspect_binary_stl(path: Path) -> None:
    data = path.read_bytes()

    if len(data) < 84:
        print(f"[LOI] File qua nho ({len(data)} bytes) - khong the la STL nhi phan hop le.")
        print("      Rat co the day la file placeholder/loi, khong phai mesh that.")
        return

    header = data[0:80]
    num_triangles = struct.unpack("<I", data[80:84])[0]
    expected_size = 84 + num_triangles * 50  # moi tam giac chiem 50 byte

    print(f"File: {path.name}")
    print(f"Kich thuoc thuc te: {len(data)} bytes")
    print(f"So tam giac khai bao trong header: {num_triangles}")
    print(f"Kich thuoc du kien neu dung STL nhi phan chuan: {expected_size} bytes")

    if len(data) != expected_size:
        print("[CANH BAO] Kich thuoc file KHONG khop voi so tam giac khai bao.")
        print("           -> Co the day la STL dang van ban (ASCII), hoac file bi loi/khong day du.")
        try:
            text_preview = data[:200].decode("utf-8", errors="ignore")
            if text_preview.strip().lower().startswith("solid"):
                print("           -> Phat hien tu khoa 'solid' o dau file: day la STL dang VAN BAN (ASCII),")
                print("              khong phai nhi phan - van la file hop le, chi khac dinh dang.")
        except Exception:
            pass
        return

    if num_triangles < 10:
        print("[CANH BAO] So tam giac qua it - kha nghi day la file placeholder/rong.")
        return

    # Tinh bounding box tu du lieu tam giac
    min_xyz = [float("inf")] * 3
    max_xyz = [float("-inf")] * 3
    offset = 84
    for _ in range(num_triangles):
        # bo qua normal vector (12 byte dau), doc 3 dinh (moi dinh 12 byte = 3 float)
        vertex_offset = offset + 12
        for v in range(3):
            x, y, z = struct.unpack("<fff", data[vertex_offset:vertex_offset + 12])
            min_xyz[0], max_xyz[0] = min(min_xyz[0], x), max(max_xyz[0], x)
            min_xyz[1], max_xyz[1] = min(min_xyz[1], y), max(max_xyz[1], y)
            min_xyz[2], max_xyz[2] = min(min_xyz[2], z), max(max_xyz[2], z)
            vertex_offset += 12
        offset += 50

    size = [max_xyz[i] - min_xyz[i] for i in range(3)]
    print(f"\nBounding box thuc te cua mesh (don vi RAW, chua biet la mm hay m):")
    print(f"  Size X: {size[0]:.3f}")
    print(f"  Size Y: {size[1]:.3f}")
    print(f"  Size Z: {size[2]:.3f}")
    print("\n-> Neu cac so nay o khoang 10-100 (vi du 50.0) va ban biet gripper that")
    print("   co kich thuoc vai cm, thi mesh nay dang o don vi MILIMET.")
    print("-> Neu cac so nay o khoang 0.01-0.1 (vi du 0.05), mesh dang o don vi MET.")
    print("[OK] File doc thanh cong, co ve la mesh that (khong phai placeholder).")


def main() -> None:
    if len(sys.argv) < 2:
        print("Cach dung: python tools/inspect_stl.py \"duong/dan/file.STL\"")
        return

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"[LOI] Khong tim thay file: {path}")
        return

    inspect_binary_stl(path)


if __name__ == "__main__":
    main()