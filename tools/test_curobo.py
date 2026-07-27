import torch

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType

from pathlib import Path

URDF_PATH = str(Path("assets/abb_irb1200_509_gripper/irb1200.urdf").resolve())
BASE_LINK = "base_link"
EE_LINK = "tool0"  # doi thanh dung ten link cuoi trong URDF neu khac


def main() -> None:
    tensor_args = TensorDeviceType()

    print(f"Dang nap URDF vao cuRobo (duong dan tuyet doi): {URDF_PATH}")
    config = CudaRobotModelConfig.from_basic_urdf(
        urdf_path=URDF_PATH,
        base_link=BASE_LINK,
        ee_link=EE_LINK,
        tensor_args=tensor_args,
    )
    robot_model = CudaRobotModel(config)

    dof = robot_model.get_dof()
    print(f"[OK] cuRobo doc URDF thanh cong. So bac tu do (DOF): {dof}")

    q_test = torch.zeros((1, dof), device=tensor_args.device, dtype=tensor_args.dtype)
    state = robot_model.get_state(q_test)

    print(f"Vi tri end-effector tai q=0: {state.ee_position}")
    print(f"Huong end-effector (quaternion wxyz) tai q=0: {state.ee_quaternion}")
    print("\n[OK] Tinh forward kinematics thanh cong - URDF hop le voi cuRobo.")


if __name__ == "__main__":
    main()