import argparse
import re
import shutil
from pathlib import Path

"""
python tools/merge_inertial_data.py --source C:/Users/BTHW/Downloads/Bang_curobo/Bang_curobo/abb/abb_irb1200_support/urdf/irb1200_5_90_macro.xacro --target assets/abb_irb1200_509_gripper/irb1200_full.urdf
"""

# Neu ten link ben nguon (ABB) khac ten link ben dich (URDF cua ban), khai o day.
# Vi du: {"link1": "link_1"} neu file goc dat ten khong co dau gach duoi.
LINK_NAME_MAP = {
    "${prefix}base_link": "base_link",
    "${prefix}link_1": "link_1",
    "${prefix}link_2": "link_2",
    "${prefix}link_3": "link_3",
    "${prefix}link_4": "link_4",
    "${prefix}link_5": "link_5", 
    "${prefix}link_6": "link_6"
}

LINK_BLOCK_PATTERN = re.compile(
    r'<link\s+name="([^"]+)"\s*>(.*?)</link>', re.DOTALL
)
INERTIAL_BLOCK_PATTERN = re.compile(r'<inertial>.*?</inertial>', re.DOTALL)


def extract_inertial_by_link(source_text: str) -> dict:
    """Doc file nguon, tra ve dict {ten_link: chuoi_khoi_<inertial>...}."""
    result = {}
    for match in LINK_BLOCK_PATTERN.finditer(source_text):
        link_name = match.group(1)
        link_body = match.group(2)
        inertial_match = INERTIAL_BLOCK_PATTERN.search(link_body)
        if inertial_match:
            mapped_name = LINK_NAME_MAP.get(link_name, link_name)
            result[mapped_name] = inertial_match.group(0)
    return result


def merge_into_target(target_text: str, inertial_by_link: dict) -> tuple:
    """Chen <inertial> vao dung <link> chua co san trong file dich."""
    inserted = []
    skipped_existing = []
    not_found_in_source = []

    def _sub(match: re.Match) -> str:
        link_name = match.group(1)
        link_body = match.group(2)

        if INERTIAL_BLOCK_PATTERN.search(link_body):
            skipped_existing.append(link_name)
            return match.group(0)  # da co san, khong dong vao

        if link_name not in inertial_by_link:
            not_found_in_source.append(link_name)
            return match.group(0)  # khong co du lieu nguon cho link nay (vd gripper)

        new_inertial = inertial_by_link[link_name]
        new_body = new_inertial + "\n" + link_body
        inserted.append(link_name)
        return f'<link name="{link_name}">{new_body}</link>'

    new_text = LINK_BLOCK_PATTERN.sub(_sub, target_text)
    return new_text, inserted, skipped_existing, not_found_in_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="File URDF/xacro goc ABB (co san <inertial>)")
    parser.add_argument("--target", required=True, help="File URDF hien tai can bo sung <inertial>")
    args = parser.parse_args()

    source_path = Path(args.source)
    target_path = Path(args.target)

    if not source_path.exists():
        print(f"[LOI] Khong tim thay file nguon: {source_path}")
        return
    if not target_path.exists():
        print(f"[LOI] Khong tim thay file dich: {target_path}")
        return

    source_text = source_path.read_text(encoding="utf-8")
    target_text = target_path.read_text(encoding="utf-8")

    inertial_by_link = extract_inertial_by_link(source_text)
    print(f"[INFO] Tim thay <inertial> cho {len(inertial_by_link)} link trong file nguon:")
    for name in inertial_by_link:
        print(f"   - {name}")

    new_text, inserted, skipped, not_found = merge_into_target(target_text, inertial_by_link)

    if not inserted:
        print("\n[CANH BAO] Khong chen duoc <inertial> nao. Kiem tra lai LINK_NAME_MAP")
        return

    backup_path = target_path.with_suffix(target_path.suffix + ".inertial.bak")
    if not backup_path.exists():
        shutil.copy2(target_path, backup_path)
        print(f"\n[BACKUP] Da luu ban goc tai: {backup_path.name}")

    target_path.write_text(new_text, encoding="utf-8")

    print(f"\n[XONG] Da chen <inertial> cho {len(inserted)} link: {inserted}")
    if skipped:
        print(f"[BO QUA] {len(skipped)} link da co san <inertial>, khong ghi de: {skipped}")
    if not_found:
        print(f"[LUU Y] {len(not_found)} link trong file dich KHONG co du lieu nguon tuong ung: {not_found}")


if __name__ == "__main__":
    main()