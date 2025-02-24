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
    private double[] pythonOutputs = new double[8];
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
            // Format for python output: llpython = [Sample Detected?, x, y, w, h, rotation, randint, randint]
            pythonOutputs = result.getPythonOutput();
            if (pythonOutputs != null && pythonOutputs.length > 0) {
                telemetry.addData("Python outputs: \n", Arrays.toString(pythonOutputs));
            }
            if (gamepad1.right_trigger > 0.5) {
                alignWithTarget(); // Align with target
            } else {
                manualDrive(); // Manual drive mode
            }
            if (gamepad1.x) limelight.pipelineSwitch(1); //Blue
            else if (gamepad1.y) limelight.pipelineSwitch(7); // Yellow
            else if (gamepad1.b) limelight.pipelineSwitch(2); //Red
            telemetry.update();
        }
    }

    private int alignWithTarget() {
        if(pythonOutputs[0] == 1){
            if (pythonOutputs[1] > 640.0) {
                robotDrive(-0.05, 0, 0);
            } else if (pythonOutputs[1] < 640.0) {
                robotDrive(0.05, 0, 0);
            }
            if (pythonOutputs[2] > 480.0) {
                robotDrive(0.0, 0.05, 0);
            } else if (pythonOutputs[2] < 480.0) {
                robotDrive(0.0, -0.05, 0);
            }
            if(pythonOutputs[0]==0){return 0;}
        }
        return 0;
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

    private void robotDrive(double x, double y, double yaw) { // to optimize and shorten robot movement in the code
        x = -x;
        double leftFrontPower = x - y - yaw;
        double rightFrontPower = x + y + yaw;
        double leftBackPower = x + y - yaw;
        double rightBackPower = x - y + yaw;

        double max = Math.max(Math.abs(leftFrontPower), Math.abs(rightFrontPower));
        max = Math.max(max, Math.abs(leftBackPower));
        max = Math.max(max, Math.abs(rightBackPower));

        leftFrontDrive.setPower(leftFrontPower / max);
        rightFrontDrive.setPower(rightFrontPower / max);
        leftBackDrive.setPower(leftBackPower / max);
        rightBackDrive.setPower(rightBackPower / max);
    }
}
