import cv2
import numpy as np

# ==================================
# VIDEO INPUT
# ==================================

video_path = "No Fill .mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video")
    exit()

# ==================================
# EMA SMOOTHING
# ==================================

alpha = 0.4
smoothed_fill = 0

# ==================================
# FILL VALIDATION
# ==================================

fill_counter = 0
required_frames = 5

# ==================================
# PROCESS VIDEO
# ==================================

while True:

    ret, img = cap.read()

    if not ret:
        print("Video ended or cannot read frame")
        break

    output = img.copy()

    # ==================================
    # MOLTEN METAL DETECTION
    # ==================================

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    v_channel = hsv[:, :, 2]

    v_channel = cv2.medianBlur(v_channel, 5)

    _, molten_mask = cv2.threshold(
        v_channel,
        220,
        255,
        cv2.THRESH_BINARY
    )

    # ==================================
    # MORPHOLOGY
    # ==================================

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )

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
    # KEEP LARGEST MOLTEN REGION
    # ==================================

    contours, _ = cv2.findContours(
        molten_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:

        largest = max(
            contours,
            key=cv2.contourArea
        )

        clean_mask = np.zeros_like(
            molten_mask
        )

        cv2.drawContours(
            clean_mask,
            [largest],
            -1,
            255,
            thickness=cv2.FILLED
        )

        molten_mask = clean_mask

    # ==================================
    # CIRCLE DETECTION
    # ==================================

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (9, 9),
        2
    )

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

        circles = np.round(
            circles[0]
        ).astype(int)

        x, y, r = max(
            circles,
            key=lambda c: c[2]
        )

        # ==================================
        # CIRCLE MASK (INVISIBLE)
        # ==================================

        circle_mask = np.zeros_like(
            molten_mask
        )

        cv2.circle(
            circle_mask,
            (x, y),
            r,
            255,
            -1
        )

        # ==================================
        # MOLTEN INSIDE CIRCLE
        # ==================================

        molten_inside = cv2.bitwise_and(
            molten_mask,
            circle_mask
        )

        # ==================================
        # FILL PERCENTAGE
        # ==================================

        circle_pixels = cv2.countNonZero(
            circle_mask
        )

        filled_pixels = cv2.countNonZero(
            molten_inside
        )

        raw_fill = (
            filled_pixels / circle_pixels
        ) * 100 if circle_pixels > 0 else 0

        # ==================================
        # EMA SMOOTHING
        # ==================================

        smoothed_fill = (
            alpha * raw_fill +
            (1 - alpha) * smoothed_fill
        )

        fill_percentage = smoothed_fill

        # ==================================
        # CONSECUTIVE FRAME VALIDATION
        # ==================================

        if fill_percentage >= 60:
            fill_counter += 1
        else:
            fill_counter = 0

        if fill_counter >= required_frames:
            status = "FILL"
            color = (0, 255, 0)
        else:
            status = "NO FILL"
            color = (0, 0, 255)

        # ==================================
        # RED MOLTEN OVERLAY
        # ==================================

        overlay = output.copy()

        overlay[molten_inside > 0] = (
            0,
            0,
            255
        )

        output = cv2.addWeighted(
            output,
            0.7,
            overlay,
            0.3,
            0
        )

        # ==================================
        # DISPLAY STATUS
        # ==================================

        cv2.putText(
            output,
            f"{status} {fill_percentage:.1f}% ({fill_counter}/{required_frames})",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            3
        )

    else:

        fill_counter = 0

        cv2.putText(
            output,
            "NO CIRCLE",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

    # ==================================
    # DISPLAY
    # ==================================

    cv2.imshow(
        "Molten Metal Detection",
        output
    )

    cv2.imshow(
        "Molten Mask",
        molten_mask
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==================================
# CLEANUP
# ==================================

cap.release()
cv2.destroyAllWindows()