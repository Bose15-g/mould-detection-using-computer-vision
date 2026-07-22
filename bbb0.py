import cv2
import numpy as np

# ==================================
# READ IMAGE
# ==================================

img = cv2.imread("WhatsApp Image 2026-04-16 at 08.52.21.jpeg")

if img is None:
    print("Image not found")
    exit()

output = img.copy()

# ==================================
# MOLTEN METAL DETECTION
# ==================================

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

_, molten_display = cv2.threshold(
    hsv,
    220,
    255,
    cv2.THRESH_BINARY
)

# Extract red channel from thresholded HSV result
molten_mask = molten_display[:, :, 2]

# Clean small noise
kernel = np.ones((5, 5), np.uint8)

molten_mask = cv2.morphologyEx(
    molten_mask,
    cv2.MORPH_OPEN,
    kernel
)

molten_mask = cv2.morphologyEx(
    molten_mask,
    cv2.MORPH_CLOSE,
    kernel
)

# ==================================
# CIRCLE DETECTION
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

if circles is not None:

    circles = np.round(circles[0]).astype(int)

    # Largest circle
    x, y, r = max(circles, key=lambda c: c[2])

    print("Center :", (x, y))
    print("Radius :", r)

    # ==================================
    # DRAW CIRCLE ONLY
    # ==================================

    cv2.circle(
        output,
        (x, y),
        r,
        (0, 255, 0),
        3
    )

    cv2.circle(
        output,
        (x, y),
        5,
        (0, 0, 0),
        -1
    )

    # ==================================
    # CREATE CIRCLE MASK
    # ==================================

    circle_mask = np.zeros_like(molten_mask)

    cv2.circle(
        circle_mask,
        (x, y),
        r,
        255,
        -1
    )

    # ==================================
    # MOLTEN METAL INSIDE CIRCLE
    # ==================================

    molten_inside = cv2.bitwise_and(
        molten_mask,
        circle_mask
    )

    # ==================================
    # FILL PERCENTAGE
    # ==================================

    circle_pixels = cv2.countNonZero(circle_mask)

    filled_pixels = cv2.countNonZero(molten_inside)

    fill_percentage = (
        filled_pixels / circle_pixels
    ) * 100

    print(
        f"Fill Percentage : {fill_percentage:.2f}%"
    )

    # ==================================
    # STATUS
    # ==================================

    if fill_percentage >= 70:
        status = "FILL"
        color = (0, 255, 0)
    else:
        status = "NO FILL"
        color = (0, 0, 255)

    # ==================================
    # RED OVERLAY FOR MOLTEN METAL
    # ==================================

    overlay = output.copy()

    overlay[molten_inside > 0] = (0, 0, 255)

    output = cv2.addWeighted(
        output,
        0.7,
        overlay,
        0.3,
        0
    )

    # Draw circle again so it stays visible
    cv2.circle(
        output,
        (x, y),
        r,
        (0, 255, 0),
        3
    )

    # ==================================
    # DISPLAY STATUS
    # ==================================

    cv2.putText(
        output,
        f"{status} {fill_percentage:.1f}%",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        color,
        3
    )

else:

    print("No circle detected")

    cv2.putText(
        output,
        "NO CIRCLE",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 0, 255),
        3
    )

# ==================================
# DISPLAY
# ==================================

cv2.imshow("Result", output)
cv2.imshow("Molten Mask", molten_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()