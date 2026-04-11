import cv2
import mediapipe as mp
import numpy as np
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.drawing_styles import get_default_hand_landmarks_style, get_default_hand_connections_style

# Initialize hand detector
mp_hands_setup = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def is_middle_finger_extended(hand_landmarks):
    """
    Check if middle finger is extended while others are folded.
    Returns True if middle finger is flipped off.
    """
    # Landmark indices
    # Fingers: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)
    # Tips: 4, 8, 12, 16, 20
    # PIP joints (base-ish): 3, 6, 10, 14, 18

    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    # Check if finger tip is ABOVE the PIP joint (for vertical orientation)
    # For middle finger (index 12 vs 10)
    middle_tip_y = hand_landmarks.landmark[12].y
    middle_pip_y = hand_landmarks.landmark[10].y

    # Check other fingers (index, ring, pinky) - they should be folded (tip BELOW pip)
    index_tip_y = hand_landmarks.landmark[8].y
    index_pip_y = hand_landmarks.landmark[6].y

    ring_tip_y = hand_landmarks.landmark[16].y
    ring_pip_y = hand_landmarks.landmark[14].y

    pinky_tip_y = hand_landmarks.landmark[20].y
    pinky_pip_y = hand_landmarks.landmark[18].y

    # Thumb is different - check horizontal position (left vs right hand logic)
    # Simplified: just check if thumb tip is to the side (not relevant for middle finger detection)

    middle_extended = middle_tip_y < middle_pip_y  # Tip above base

    others_folded = (
        index_tip_y > index_pip_y and      # Index folded
        ring_tip_y > ring_pip_y and        # Ring folded
        pinky_tip_y > pinky_pip_y          # Pinky folded
    )

    return middle_extended and others_folded

def get_hand_bounding_box(hand_landmarks, image_shape):
    """Get bounding box around the hand for censoring."""
    h, w, _ = image_shape
    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
    y_coords = [lm.y * h for lm in hand_landmarks.landmark]

    x_min = int(min(x_coords)) - 20
    x_max = int(max(x_coords)) + 20
    y_min = int(min(y_coords)) - 30
    y_max = int(max(y_coords)) + 30

    # Keep within image bounds
    x_min = max(0, x_min)
    x_max = min(w, x_max)
    y_min = max(0, y_min)
    y_max = min(h, y_max)

    return (x_min, y_min, x_max, y_max)

def apply_censor(image, bbox, censor_type="black_bar"):
    """Apply censorship (black bar or blur) to the hand region."""
    x_min, y_min, x_max, y_max = bbox

    if censor_type == "black_bar":
        # Draw black rectangle over the hand
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), -1)

        # Optional: Add text
        cv2.putText(image, "CENSORED", (x_min + 10, y_min + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    elif censor_type == "blur":
        # Extract hand region, blur it, put it back
        hand_region = image[y_min:y_max, x_min:x_max]
        if hand_region.size > 0:
            blurred = cv2.GaussianBlur(hand_region, (55, 55), 30)
            image[y_min:y_max, x_min:x_max] = blurred

            # Add text
            cv2.putText(image, "BLURRED", (x_min + 10, y_min + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return image

# Start webcam
cap = cv2.VideoCapture(0)
censor_style = "black_bar"  # Change to "blur" if you prefer blurring

print("🚨 Middle Finger Detector Active! 🚨")
print("Press 'q' to quit")
print("Press 'b' to toggle between black bar and blur")
print("Press 'f' to toggle censorship on/off")

censor_enabled = True

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip for mirror view, convert to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = mp_hands_setup.process(image)

    # Convert back to BGR for OpenCV
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Check if this hand is flipping off
            if is_middle_finger_extended(hand_landmarks):
                # Get bounding box and censor
                bbox = get_hand_bounding_box(hand_landmarks, image.shape)
                if censor_enabled:
                    image = apply_censor(image, bbox, censor_style)

                # Show warning text
                cv2.putText(image, "⚠️  MIDDLE FINGER DETECTED!  ⚠️",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 3)
            else:
                # Draw normal hand landmarks (optional)
                draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    get_default_hand_landmarks_style(),
                    get_default_hand_connections_style()
                )

            # Show handedness
            label = f"{handedness.classification[0].label}"
            cv2.putText(image, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show status text
    status = f"Censor: {'ON' if censor_enabled else 'OFF'} | Style: {censor_style}"
    cv2.putText(image, status, (10, image.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('Middle Finger Detector & Censor', image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):
        # Toggle between black bar and blur
        censor_style = "blur" if censor_style == "black_bar" else "black_bar"
        print(f"Censor style changed to: {censor_style}")
    elif key == ord('f'):
        censor_enabled = not censor_enabled
        print(f"Censorship: {'ON' if censor_enabled else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
