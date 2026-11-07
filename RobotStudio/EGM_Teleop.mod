MODULE EGM_Teleop
    ! TCP = 161 mm along tool0 z, same as robotiq_tcp in the URDF used by cuRobo.
    ! Mass / center of gravity are placeholders, set the real gripper values.
    PERS tooldata tGripper := [TRUE,[[0,0,161],[1,0,0,0]],[1,[0,0,60],[1,0,0,0],0,0,0]];

    ! Same home as cuRobo (0,0,0,0,90,90 deg)
    CONST jointtarget jHome := [[0,0,0,0,90,90],[9E9,9E9,9E9,9E9,9E9,9E9]];

    VAR egmident egmID1;
    VAR egmstate egmSt1;

    ! Convergence criteria (mm / deg) used by EGMRunPose
    CONST egm_minmax egm_minmax_lin1 := [-1,1];
    CONST egm_minmax egm_minmax_rot1 := [-2,2];

    ! Identity frames: references are absolute TCP poses in wobj0 (robot base)
    VAR pose posecor := [[0,0,0],[1,0,0,0]];
    VAR pose posesen := [[0,0,0],[1,0,0,0]];

    PROC main()
        MoveAbsJ jHome, v200, fine, tGripper;

        EGMReset egmID1;
        EGMGetId egmID1;
        egmSt1 := EGMGetState(egmID1);
        IF egmSt1 <= EGM_STATE_CONNECTED THEN
            ! "default" = External Motion Interface Data instance, "UCdevice" = UDPUC entry (127.0.0.1:6510)
            EGMSetupUC ROB_1, egmID1, "default", "UCdevice" \Pose \CommTimeout:=60;
        ENDIF

        ! MaxSpeedDeviation (deg/s) caps the joint speed EGM may use to follow the reference.
        ! Left at its default the robot moved at ~1 deg/s in the logs. Start low on the real robot.
        ! EGMRunPose returns after the robot has converged for CondTime, so loop it
        WHILE TRUE DO
            EGMActPose egmID1 \Tool:=tGripper \WObj:=wobj0, posecor, EGM_FRAME_WOBJ, posesen, EGM_FRAME_WOBJ
                \x:=egm_minmax_lin1 \y:=egm_minmax_lin1 \z:=egm_minmax_lin1
                \rx:=egm_minmax_rot1 \ry:=egm_minmax_rot1 \rz:=egm_minmax_rot1
                \LpFilter:=20 \MaxSpeedDeviation:=60;
            EGMRunPose egmID1, EGM_STOP_HOLD \x \y \z \rx \ry \rz \CondTime:=10 \RampInTime:=0.05;
        ENDWHILE
    ENDPROC
ENDMODULE
