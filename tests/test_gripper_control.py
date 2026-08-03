from src.worker_pybullet import compute_gripper_control


def test_close_command_stops_when_contact_detected():
    target, force, holding = compute_gripper_control(
        current_pos=0.16,
        gripper_target=1.0,
        contact_detected=True,
    )

    assert target == 0.16
    assert force <= 35
    assert holding is True


def test_close_command_advances_slowly_without_contact():
    target, force, holding = compute_gripper_control(
        current_pos=0.16,
        gripper_target=1.0,
        contact_detected=False,
    )

    assert target == 0.17
    assert force <= 35
    assert holding is False
