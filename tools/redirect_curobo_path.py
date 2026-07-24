import re
import shutil
from pathlib import Path

# ---- CAU HINH: chinh lai neu cau truc thu muc cua ban khac ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # tools/ -> project root
ROBOT_ASSET_DIR = PROJECT_ROOT / "assets" / "abb_irb1200_509_gripper"

URDF_FILES = [
    ROBOT_ASSET_DIR / "irb1200.urdf",
    ROBOT_ASSET_DIR / "irb1200_full.urdf",
]

ARM_MESH_DIR = "meshes_robot"
GRIPPER_MESH_DIR = "meshes_gripper"  

ARM_KEYWORDS = ["irb1200", "abb"]

GRIPPER_RENAME_MAP = {
    "robotiq_85_base_link_fine.stl":  ("visual",    "robotiq_arg2f_85_base_link.dae"),
    "robotiq_85_base_link_coarse.stl": ("collision", "robotiq_arg2f_85_base_link.stl"),
    "outer_knuckle_fine.stl":   ("visual",    "robotiq_arg2f_85_outer_knuckle.dae"),
    "outer_knuckle_coarse.stl": ("collision", "robotiq_arg2f_85_outer_knuckle.dae"),
    "inner_knuckle_fine.stl":   ("visual",    "robotiq_arg2f_85_inner_knuckle.dae"),
    "inner_knuckle_coarse.stl": ("collision", "robotiq_arg2f_85_inner_knuckle.dae"),
    "outer_finger_fine.stl":    ("visual",    "robotiq_arg2f_85_outer_finger.dae"),
    "outer_finger_coarse.stl":  ("collision", "robotiq_arg2f_85_outer_finger.dae"),
    "inner_finger_fine.stl":    ("visual",    "robotiq_arg2f_85_inner_finger.dae"),
    "inner_finger_coarse.stl":  ("collision", "robotiq_arg2f_85_inner_finger.dae"),
}

MESH_PATTERN = re.compile(r'filename="([^"]+)"')


def classify_and_rewrite(old_path: str) -> str:
    """Nhan path tuyet doi cu, tra ve path tuong doi moi (dung '/')."""
    normalized = old_path.replace("\\", "/")
    lower = normalized.lower()
    basename = normalized.rstrip("/").split("/")[-1]
    basename_lower = basename.lower()

    # Uu tien 1: gripper - dung bang mapping thu cong vi ten file khac hoan toan
    if basename_lower in GRIPPER_RENAME_MAP:
        subfolder, real_name = GRIPPER_RENAME_MAP[basename_lower]
        return f"{GRIPPER_MESH_DIR}/{subfolder}/{real_name}"

    # Uu tien 2: tay may ABB - van dung heuristic theo ten thu muc cha
    if "/collision/" in lower:
        subfolder = "collision"
    elif "/visual/" in lower:
        subfolder = "visual"
    else:
        subfolder = "visual"  # fallback, hiem khi xay ra voi mesh tay may

    return f"{ARM_MESH_DIR}/{subfolder}/{basename}"


def process_file(urdf_path: Path) -> None:
    if not urdf_path.exists():
        print(f"[BO QUA] Khong tim thay file: {urdf_path}")
        return

    text = urdf_path.read_text(encoding="utf-8")
    replaced = []

    def _sub(match: re.Match) -> str:
        old = match.group(1)
        new = classify_and_rewrite(old)
        replaced.append((old, new))
        return f'filename="{new}"'

    new_text = MESH_PATTERN.sub(_sub, text)

    if not replaced:
        print(f"[CANH BAO] Khong tim thay dong 'filename=\"...\"' nao trong {urdf_path.name}")
        return

    # Backup file goc truoc khi ghi de (chi backup 1 lan)
    backup_path = urdf_path.with_suffix(urdf_path.suffix + ".bak")
    if not backup_path.exists():
        shutil.copy2(urdf_path, backup_path)
        print(f"[BACKUP] Da luu ban goc tai: {backup_path.name}")
    else:
        print(f"[BACKUP] Da co san ({backup_path.name}), khong ghi de.")

    urdf_path.write_text(new_text, encoding="utf-8")
    print(f"[XONG] Da cap nhat {len(replaced)} duong dan mesh trong {urdf_path.name}")

    # Kiem tra mesh co thuc su ton tai tai vi tri moi khong
    base_dir = urdf_path.parent
    missing = []
    for old, new in replaced:
        target = base_dir / new
        if not target.exists():
            missing.append((old, new))

    if missing:
        print(f"\n[CANH BAO] {len(missing)} mesh KHONG tim thay sau khi doi path:")
        for old, new in missing:
            print(f"   - {new}")
            print(f"     (path cu: {old})")
        print("   -> Kiem tra lai: ten file co dung khong, hoa/thuong co khop khong,")
        print("      hoac chua copy mesh vao dung thu muc assets/robot/... chua.\n")
    else:
        print(f"[OK] Tat ca {len(replaced)} mesh deu duoc tim thay dung vi tri moi.\n")


def main() -> None:
    print(f"Thu muc asset robot: {ROBOT_ASSET_DIR}\n")
    for urdf_path in URDF_FILES:
        process_file(urdf_path)


if __name__ == "__main__":
    main()