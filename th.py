import cv2

img = cv2.imread("WhatsApp Image 2026-04-16 at 08.52.21.jpeg")

if img is None:
    print("Image not found")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

_, molten_mask = cv2.threshold(
    gray,
    220,
    255,
    cv2.THRESH_BINARY
)

cv2.imshow("Original", img)
cv2.imshow("Molten Metal", molten_mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
#ADD CIRCLE DETECTION TO FIND THE RIM