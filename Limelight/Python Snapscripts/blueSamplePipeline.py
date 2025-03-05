import cv2
import numpy as np
COLOR_BOUNDS = {
    "yellow": [(12, 240, 57), (38, 255, 230)],
    "red1": [(0, 222, 23), (9, 255, 243)],
    "red2": [(160, 222, 23), (255, 255, 243)],
    "blue": [(110, 240, 0), (130, 255, 225)],
}
# Function to draw decorations (text) on the image
def drawDecorations(image, color_name, hsv_values, mf='N/A'):
    # Display the detected color name
    cv2.putText(image,
        f'Color: {color_name}',
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2, cv2.LINE_AA)
    # Display the HSV values
    cv2.putText(image,
        f'HSV: {hsv_values}',
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2, cv2.LINE_AA)
    #Show orientation
    cv2.putText(image,
        f'Orientiation: {mf}',
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2, cv2.LINE_AA)
# runPipeline() is called every frame by Limelight's backend.
def runPipeline(image, llrobot):
    # Convert the image to HSV
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define the HSV range for the color you want to detect (e.g., green)
    lower_hsv = COLOR_BOUNDS['blue'][0]
    upper_hsv = COLOR_BOUNDS['blue'][1]
    kernel = np.ones((9,9), dtype=np.uint8)
    img_threshold = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
    eroded = cv2.erode(img_threshold, kernel, iterations=1)
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largestContour = np.array([[]])
    llpython = [0, 0, 0, 0, 0, 0, 0, 0]
    color_name = "blue"
    hsv_values = f"{upper_hsv}"
    if len(contours) > 0:
        # Find the largest contour
        largestContour = max(contours, key=cv2.contourArea)
        # Approximate the contour with more points
        epsilon = 0.01 * cv2.arcLength(largestContour, True)  # Lower value = more points
        approx = cv2.approxPolyDP(largestContour, epsilon, True)
        # Get a convex hull for smooth contouring
        hull = cv2.convexHull(approx)
        # Get bounding box parameters
        erect = cv2.minAreaRect(largestContour)
        (x, y), (w, h), angle = erect
        if 20 < w < 1000:
            # Draw a more accurate contour instead of a box
            cv2.polylines(image, [hull], isClosed=True, color=(0, 0, 255), thickness=25)
            cv2.circle(image, (int(x), int(y)), color=(0, 0, 255), radius=2, thickness=2)
            if w < h:
                angle += 90
            drawDecorations(image, color_name, hsv_values, angle)
            llpython = [1 if w > 0 else 0, x, y, w, h, angle, 8, 6]
    drawDecorations(image, color_name, hsv_values, '')
    return largestContour, image, llpython 
