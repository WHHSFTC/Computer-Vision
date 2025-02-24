package org.firstinspires.ftc.teamcode;

import com.qualcomm.hardware.limelightvision.Limelight3A;
import com.qualcomm.robotcore.eventloop.opmode.Disabled;
import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.eventloop.opmode.TeleOp;
import com.qualcomm.robotcore.hardware.DcMotor;
import com.qualcomm.hardware.limelightvision.LLResult;
import com.qualcomm.hardware.limelightvision.LLResultTypes;
import com.qualcomm.hardware.limelightvision.LLStatus;
import edu.wpi.first.networktables.NetworkTable;
import edu.wpi.first.networktables.NetworkTableInstance;



@TeleOp(name = "Limelight Sample Approach w/ manual drive")
// @Disabled

public class SensorLimelight3A extends LinearOpMode {
    private Limelight3A limelight;
    private double tx, ty, ta, tv; // Vision variables
    private double targetAlignThreshold = 10.0;
    private double moveSpeed = 0.2;
    private DcMotor leftFrontDrive, rightFrontDrive, leftBackDrive, rightBackDrive;

    @Override
    public void runOpMode() throws InterruptedException {
        // Initialize Limelight and motors
        limelight = hardwareMap.get(Limelight3A.class, "limelight");
        limelight.pipelineSwitch(5);

        leftFrontDrive = hardwareMap.get(DcMotor.class, "frontLeft");
        rightFrontDrive = hardwareMap.get(DcMotor.class, "frontRight");
        leftBackDrive = hardwareMap.get(DcMotor.class, "backLeft");
        rightBackDrive = hardwareMap.get(DcMotor.class, "backRight");

        leftFrontDrive.setDirection(DcMotor.Direction.FORWARD);
        leftBackDrive.setDirection(DcMotor.Direction.FORWARD);
        rightFrontDrive.setDirection(DcMotor.Direction.REVERSE);
        rightBackDrive.setDirection(DcMotor.Direction.REVERSE);

        waitForStart();

        limelight.start();

        while (opModeIsActive()) {
            // Get the latest data from Limelight
            LLResult result = limelight.getLatestResult();

            if (result != null) {
                tv = result.getTyNC();
                if (tv != 0) tv = 1;
                else tv = 0;
                tx = result.getTx(); // Horizontal offset
                ty = result.getTy(); // Vertical offset
                ta = result.getTa(); // Target area
            } else { //default to 0 to prevent unwanted movement
                tv = 0.0;
                tx = 0.0;
                ty = 0.0;
                ta = 0.0;
            }
            // Display telemetry
            telemetry.addData("TX", tx); // Lines 47-49
            telemetry.addData("TY", ty);
            telemetry.addData("TA", ta);
            telemetry.addData("TV", tv);
            telemetry.update();
            // Right trigger to activate auto-align
            if (gamepad1.right_trigger > 0.5) {
                if (tv == 1.0) {
                    alignWithTarget(); // Align with target
                } else {
                    stopRobot(); // No target found
                }
            } else {
                manualDrive(); // Manual drive mode
            }
        }

        limelight.stop();

    }

    private void alignWithTarget() {
        if (tx > targetAlignThreshold) {
            robotDrive(moveSpeed, -moveSpeed); // Turn right
        } else if (tx < -targetAlignThreshold) {
            robotDrive(-moveSpeed, moveSpeed); // Turn left
        } else {
            robotDrive(-moveSpeed, -moveSpeed);
        }
    }

    private void manualDrive() {
        double drive = -gamepad1.left_stick_y;  // Forward/reverse
        double strafe = gamepad1.left_stick_x; // Strafing
        double yaw = gamepad1.right_stick_x;   // Turning
        double leftFrontPower = drive + strafe + yaw;
        double rightFrontPower = drive - strafe - yaw;
        double leftBackPower = drive - strafe + yaw;
        double rightBackPower = drive + strafe - yaw;
        double max = Math.max(1.0, Math.max(
                Math.max(Math.abs(leftFrontPower), Math.abs(rightFrontPower)),
                Math.max(Math.abs(leftBackPower), Math.abs(rightBackPower))
        ));
        leftFrontDrive.setPower(leftFrontPower / max);
        rightFrontDrive.setPower(rightFrontPower / max);
        leftBackDrive.setPower(leftBackPower / max);
        rightBackDrive.setPower(rightBackPower / max);
    }

    private void robotDrive(double leftPower, double rightPower) {
        leftFrontDrive.setPower(leftPower);
        rightFrontDrive.setPower(rightPower);
        leftBackDrive.setPower(leftPower);
        rightBackDrive.setPower(rightPower);
    }

    private void stopRobot() {
        robotDrive(0, 0); // Stop all motion
    }
}
