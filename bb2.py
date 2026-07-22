import cv2
import numpy as np

# ==================================
# IMAGE INPUT
# ==================================

img = cv2.imread("WhatsApp Image 2026-04-16 at 08.52.21.jpeg")

if img is None:
    print("Image not found")
    exit()

output = img.copy()

# ==================================
# SMOOTHING VARIABLE
# ==================================

alpha = 0.3
smoothed_fill = 0

# ==================================
# MOLTEN METAL DETECTION
# ==================================

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

_, molten_mask = cv2.threshold(
    hsv[:, :, 2],
    220,
    255,
    cv2.THRESH_BINARY
)

# Morphology cleanup
kernel = np.ones((5, 5), np.uint8)
molten_mask = cv2.morphologyEx(molten_mask, cv2.MORPH_OPEN, kernel)
molten_mask = cv2.morphologyEx(molten_mask, cv2.MORPH_CLOSE, kernel)

# ==================================
# CIRCLE DETECTION (RIM)
# ==================================

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (9, 9), 2)

circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=100,
    param1=50,
    param2=30,
    minRadius=100,
    maxRadius=500
)

# ==================================
# PROCESS IF CIRCLE FOUND
# ==================================

if circles is not None:

    circles = np.round(circles[0]).astype(int)
    x, y, r = max(circles, key=lambda c: c[2])

    # Draw rim circle
    cv2.circle(output, (x, y), r, (0, 255, 0), 3)

    # ==================================
    # MASK INSIDE CIRCLE
    # ==================================

    circle_mask = np.zeros_like(molten_mask)
    cv2.circle(circle_mask, (x, y), r, 255, -1)

    molten_inside = cv2.bitwise_and(molten_mask, circle_mask)

    # ==================================
    # FILL CALCULATION
    # ==================================

    circle_pixels = cv2.countNonZero(circle_mask)
    filled_pixels = cv2.countNonZero(molten_inside)

    raw_fill = (filled_pixels / circle_pixels) * 100 if circle_pixels > 0 else 0

    # ==================================
    # SMOOTHING (EXPO MOVING AVERAGE)
    # ==================================

    smoothed_fill = alpha * raw_fill + (1 - alpha) * smoothed_fill

    fill_percentage = smoothed_fill

    # ==================================
    # STATUS
    # ==================================

    if fill_percentage >= 60:
        status = "FILL"
        color = (0, 255, 0)
    else:
        status = "NO FILL"
        color = (0, 0, 255)

    # ==================================
    # VISUALIZATION
    # ==================================

    overlay = output.copy()
    overlay[molten_inside > 0] = (0, 0, 255)

    output = cv2.addWeighted(output, 0.7, overlay, 0.3, 0)

    cv2.putText(
        output,
        f"{status} {fill_percentage:.1f}%",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3
    )

else:
    cv2.putText(
        output,
        "NO CIRCLE DETECTED",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3
    )

# ==================================
# DISPLAY
# ==================================

cv2.imshow("Original", output)
cv2.imshow("Molten Mask", molten_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()