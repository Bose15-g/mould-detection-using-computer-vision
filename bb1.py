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
# DETECT RED MOLTEN METAL
# ==================================

# Red color range (BGR)
lower_red = np.array([0, 0, 200], dtype=np.uint8)
upper_red = np.array([80, 80, 255], dtype=np.uint8)

molten_mask = cv2.inRange(img, lower_red, upper_red)

# Remove small noise
kernel = np.ones((5, 5), np.uint8)
molten_mask = cv2.morphologyEx(
    molten_mask,
    cv2.MORPH_OPEN,
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

    # Draw circle boundary in BLACK
    cv2.circle(output, (x, y), r, (0, 0, 0), 3)

    # Draw center point in BLACK
    cv2.circle(output, (x, y), 6, (0, 0, 0), -1)

    # Rim line
    rim_y = y - r

    cv2.line(
        output,
        (0, rim_y),
        (output.shape[1], rim_y),
        (255, 0, 0),
        2
    )

    # ==================================
    # CIRCLE MASK
    # ==================================

    circle_mask = np.zeros(
        molten_mask.shape,
        dtype=np.uint8
    )

    cv2.circle(
        circle_mask,
        (x, y),
        r,
        255,
        -1
    )

    # ==================================
    # RED PIXELS INSIDE CIRCLE
    # ==================================

    red_inside = cv2.bitwise_and(
        molten_mask,
        molten_mask,
        mask=circle_mask
    )

    red_pixels = cv2.countNonZero(red_inside)
    total_pixels = cv2.countNonZero(circle_mask)

    fill_percentage = (
        red_pixels / total_pixels
    ) * 100

    # ==================================
    # DECISION
    # ==================================

    if fill_percentage >= 70:
        status = "FILL"
        status_color = (0, 255, 0)
    else:
        status = "NO FILL"
        status_color = (0, 0, 255)

    # ==================================
    # DRAW RESULTS
    # ==================================

    cv2.putText(
        output,
        f"Fill: {fill_percentage:.1f}%",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        2
    )

    cv2.putText(
        output,
        status,
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        status_color,
        3
    )

    print("--------------------------------")
    print("Center :", (x, y))
    print("Radius :", r)
    print("Rim Y  :", rim_y)
    print(f"Fill Percentage : {fill_percentage:.2f}%")
    print("Status :", status)
    print("--------------------------------")

else:
    print("No circle detected")

# ==================================
# DISPLAY
# ==================================

cv2.imshow("Original + Result", output)
cv2.imshow("Molten Metal Mask", molten_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()