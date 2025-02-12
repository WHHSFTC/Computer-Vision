package org.firstinspires.ftc.teamcode;

import java.util.*;

import com.qualcomm.hardware.limelightvision.Limelight3A;
import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.eventloop.opmode.TeleOp;
import com.qualcomm.hardware.limelightvision.LLResult;
import com.qualcomm.hardware.limelightvision.LLResultTypes;
import com.qualcomm.robotcore.hardware.DcMotor;
import com.qualcomm.robotcore.hardware.Servo;

import java.util.List;

@TeleOp(name = "SnapScript Pipeline")
public class snapTism extends LinearOpMode {

    private Limelight3A limelight;
    private double tx, ty; // Vision variables
    private double targetAlignThreshold = 10.0;
    private double moveSpeed = 0.2;
    private double[] pythonOutputs = new double[32];
    private DcMotor leftFrontDrive, rightFrontDrive, leftBackDrive, rightBackDrive;
    private Servo wrist;

    @Override
    public void runOpMode() throws InterruptedException {
        limelight = hardwareMap.get(Limelight3A.class, "limelight");
        limelight.setPollRateHz(50);
        limelight.start();

        leftFrontDrive = hardwareMap.get(DcMotor.class, "frontLeft");
        rightFrontDrive = hardwareMap.get(DcMotor.class, "frontRight");
        leftBackDrive = hardwareMap.get(DcMotor.class, "backLeft");
        rightBackDrive = hardwareMap.get(DcMotor.class, "backRight");

        leftFrontDrive.setDirection(DcMotor.Direction.FORWARD);
        leftBackDrive.setDirection(DcMotor.Direction.FORWARD);
        rightFrontDrive.setDirection(DcMotor.Direction.REVERSE);
        rightBackDrive.setDirection(DcMotor.Direction.REVERSE);

        wrist = hardwareMap.servo.get("wristServo"); // Port 1

        waitForStart();
        while (opModeIsActive()) {
            LLResult result = limelight.getLatestResult();
            // Getting numbers from Python
            // Format for python output: llpython = [Sample Detected?, x, y, w, h, Orientation (1 for H, 0 for V), randint, randint]
            pythonOutputs = result.getPythonOutput();
            if (pythonOutputs != null && pythonOutputs.length > 0) {
                telemetry.addData("Python outputs: \n", Arrays.toString(pythonOutputs));
            }
            if (gamepad1.right_trigger > 0.5) {
                alignWithTarget(); // Align with target
                telemetry.addData("IN IF", 0);
            } else {
                manualDrive(); // Manual drive mode
            }
            if (gamepad1.x) limelight.pipelineSwitch(1);
            else if (gamepad1.y) limelight.pipelineSwitch(7);
            else if (gamepad1.b) limelight.pipelineSwitch(2);
            telemetry.update();
        }
    }

    private void alignWithTarget() {
        if (tx > targetAlignThreshold) {
            robotDrive(moveSpeed, -moveSpeed); // Turn right
        } else if (tx < -targetAlignThreshold) {
            robotDrive(-moveSpeed, moveSpeed); // Turn lefts
        } else {
            robotDrive(-moveSpeed, -moveSpeed);
        }
        if (pythonOutputs[5] == 0.0) {
            telemetry.addData("IN ALIGN", .5);
            wrist.setPosition(0.5);
        }else if (pythonOutputs[5] == 1.0){
            telemetry.addData("IN ALIGN", 0);
            wrist.setPosition(0.2);
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