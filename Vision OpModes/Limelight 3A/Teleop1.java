package org.firstinspires.ftc.teamcode;

import java.util.*;
import com.qualcomm.hardware.limelightvision.Limelight3A;
import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.eventloop.opmode.TeleOp;
import com.qualcomm.hardware.limelightvision.LLResult;
import com.qualcomm.hardware.limelightvision.LLResultTypes;

import java.util.List;

@TeleOp(name = "colorPipeline")
public class Teleop1 extends LinearOpMode {

    Limelight3A limelight;

    @Override
    public void runOpMode() throws InterruptedException {
        limelight = hardwareMap.get(Limelight3A.class, "limelight");
        limelight.setPollRateHz(100);
        limelight.start();
        limelight.pipelineSwitch(1); // Ensure this is called AFTER start()
        waitForStart();
        while (opModeIsActive()) {
            LLResult result = limelight.getLatestResult();
            if (result != null && result.isValid()) {
                double tx = result.getTx();
                double ty = result.getTy();
                double ta = result.getTa();

                telemetry.addData("Target X", tx);
                telemetry.addData("Target Y", ty);
                telemetry.addData("Target Area", ta);
            } else {
                telemetry.addData("Limelight", "No Targets");
            }
            // Check if result is null before accessing getColorResults
            if (result != null) {
                List<LLResultTypes.ColorResult> colors = result.getColorResults();
                if (colors != null) {
                    for (LLResultTypes.ColorResult color : colors) {
                        double x = color.getTargetXDegrees();
                        double y = color.getTargetYDegrees();
                        List<List<Double>> endColors = color.getTargetCorners();
                        telemetry.addData("x: ", x);
                        telemetry.addData("y: ", y);
                        telemetry.addData("CornersList", endColors);
                        telemetry.addData("Angle thingy",findShortestSide(endColors));
                    }
                }
            }

            telemetry.update();
        }
    }
    public static double findShortestSide(List<List<Double>> points) {
        if (points.size() < 2) {
            System.out.println("Not enough points to form a shape.");
            return 0;
        }

        double minDistance = Double.MAX_VALUE;
        double angle = 0.0;
        List<Double> shortestStart = new ArrayList<>();
        List<Double> shortestEnd = new ArrayList<>();

        for (int i = 0; i < points.size(); i++) {
            List<Double> p1 = points.get(i);
            List<Double> p2 = points.get((i + 1) % points.size()); // Wrap around

            double dx = p2.get(0) - p1.get(0);
            double dy = p2.get(1) - p1.get(1);
            double distance = Math.sqrt(dx * dx + dy * dy);

            if (distance < minDistance) {
                minDistance = distance;
                angle = Math.toDegrees(Math.atan2(dy, dx)); // Calculate angle in degrees
                shortestStart = p1;
                shortestEnd = p2;
            }
        }
        return angle;
    }
}