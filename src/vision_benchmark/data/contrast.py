"""How much a defect stands out from its surroundings, split into brightness and color.

Uses CIE Lab: L is brightness (what a grayscale image keeps), a and b are color (what grayscale drops).
If delta_L is close to delta_E, the defect is visible without color.
"""

import cv2
import numpy as np


def defect_contrast(img_bgr, mask, ring_px=15):
    """Contrast between defect pixels (mask > 0) and a ring of ring_px pixels around them.

    Returns delta_L (brightness), delta_ab (color only) and delta_E (both), in Lab units.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * 100.0 / 255.0  # OpenCV stores 8-bit L as 0..255, Lab units are 0..100
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0

    defect = mask > 0
    if not defect.any():
        raise ValueError("empty mask")
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
    ring = (cv2.dilate(defect.astype(np.uint8), k) > 0) & ~defect

    dL = L[defect].mean() - L[ring].mean()
    da = a[defect].mean() - a[ring].mean()
    db = b[defect].mean() - b[ring].mean()
    return dict(delta_L=float(abs(dL)), delta_ab=float(np.hypot(da, db)),
                delta_E=float(np.sqrt(dL ** 2 + da ** 2 + db ** 2)))
