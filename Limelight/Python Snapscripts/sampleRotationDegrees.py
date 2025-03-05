import cv2
import numpy as np
import math
verticalBias = 1.5
def runPipeline(image, llrobot):
    # Initialize variables to avoid potential unboundlocal errors
    longest_line = None
    max_length = 0
    angle = 0
    llrobot = [180, 150, 280, 90] # for debugging
    # Define the Region of Interest (ROI)
    # We'll use the first 4 values from llrobot as [x, y, width, height]
    # Ensure these values are within the image bounds
    height, width = image.shape[:2]
    roi_x = max(0, min(int(llrobot[0]), width - 1))
    roi_y = max(0, min(int(llrobot[1]), height - 1))
    roi_w = max(1, min(int(llrobot[2]), width - roi_x))
    roi_h = max(1, min(int(llrobot[3]), height - roi_y))
    # Extract the ROI
    roi = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    # Preprocess the ROI
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)  # Reduce noise
    edges = cv2.Canny(blurred_roi, 100, 90)
    # Perform Hough Line Detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=15, maxLineGap=20)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2 - x1)**2 + (verticalBias * y2 - verticalBias * y1)**2)
            if length > max_length:
                max_length = length
                longest_line = line[0]
        if longest_line is not None:
            x1, y1, x2, y2 = longest_line
            # Calculate angle in degrees
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            # Ensure angle is between 0 and 180 degrees
            angle = angle - 90 if angle >= 0 else angle + 90
            # Draw the longest line on the ROI
            cv2.line(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Draw the ROI rectangle on the main image
    cv2.rectangle(image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
    # Add text with angle information
    cv2.putText(image, f"Angle: {angle:.2f} degrees", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Prepare the output array
    llpython = [angle, 0, 0, 0, 0, 0, 0, 0]
    # Return an empty contour (not used in this case), the processed image, and the llpython array
    return np.array([[]]), image, llpython
