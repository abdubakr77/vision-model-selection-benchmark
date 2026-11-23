"""Board orientation check for VisA PCB images ("upright" or "rotated_180").

Two independent signals, no labels used:
- crystal: the silver crystal sits in the top part of the board when upright, bottom when rotated 180
- pins: the pin header sticks out below the board when upright, above when rotated 180
If they disagree the image should be checked by eye.
"""

import cv2
import numpy as np

# HSV range of the blue PCB (OpenCV hue is 0..179)
BOARD_HSV_LOW = (85, 80, 40)
BOARD_HSV_HIGH = (130, 255, 255)


def find_board(img_bgr):
    """Bounding box (x0, y0, x1, y1) of the largest blue region, which is the board."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BOARD_HSV_LOW, BOARD_HSV_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        raise ValueError("no blue board found")
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[i, :4]
    return int(x), int(y), int(x + w), int(y + h)


def detect_orientation(img_bgr):
    """Return a dict with orientation ("upright" or "rotated_180"), both votes, agreement and raw scores."""
    x0, y0, x1, y1 = find_board(img_bgr)
    bw, bh = x1 - x0, y1 - y0

    # crystal vote: share of bright, low saturation pixels in the top vs bottom 30% of the central columns
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    band = hsv[y0:y1, x0 + int(0.38 * bw):x0 + int(0.62 * bw)]
    silver = ((band[..., 1] < 60) & (band[..., 2] > 150)).astype(np.float32)
    k = max(1, int(0.3 * bh))
    crystal_top, crystal_bottom = float(silver[:k].mean()), float(silver[-k:].mean())

    # pins vote: vertical edge strength in a strip just above vs just below the board
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edges = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    s = int(0.3 * bh)
    cols = slice(x0 + int(0.3 * bw), x0 + int(0.7 * bw))
    above_strip = edges[max(0, y0 - s):max(0, y0 - 5), cols]
    below_strip = edges[min(gray.shape[0], y1 + 5):min(gray.shape[0], y1 + s), cols]
    pins_above = float(above_strip.mean()) if above_strip.size else 0.0
    pins_below = float(below_strip.mean()) if below_strip.size else 0.0

    crystal_vote = "upright" if crystal_top > crystal_bottom else "rotated_180"
    pins_vote = "upright" if pins_below > pins_above else "rotated_180"
    return dict(orientation=pins_vote if crystal_vote == pins_vote else "unsure",
                crystal_vote=crystal_vote, pins_vote=pins_vote, agree=crystal_vote == pins_vote,
                crystal_top=crystal_top, crystal_bottom=crystal_bottom,
                pins_above=pins_above, pins_below=pins_below, board=(x0, y0, x1, y1))
