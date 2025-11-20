
import numpy as np
import cv2
from scipy.ndimage.interpolation import rotate
import time

def test_rotation():
    # Create a dummy image (simulating CT slice)
    # Shape 208x208, int16, range -1000 to 1000
    img = np.random.randint(-1000, 1000, (208, 208), dtype=np.int16)
    
    angle = 15
    AIR_HU_VAL = -1000.0
    
    # Scipy rotation
    t0 = time.time()
    scipy_out = rotate(img, angle, axes=(-2, -1), reshape=False, cval=AIR_HU_VAL)
    t1 = time.time()
    print(f"Scipy time: {t1-t0:.4f}s")
    
    # CV2 rotation
    t0 = time.time()
    height, width = img.shape
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cv2_out = cv2.warpAffine(img, M, (width, height), borderValue=AIR_HU_VAL)
    t1 = time.time()
    print(f"CV2 time: {t1-t0:.4f}s")
    
    # Compare
    # Note: Interpolation differences are expected, but should be correlated
    diff = np.abs(scipy_out.astype(float) - cv2_out.astype(float))
    print(f"Mean diff: {diff.mean()}")
    print(f"Max diff: {diff.max()}")
    
    # Check if CV2 output is all border value (bug check)
    if np.all(cv2_out == int(AIR_HU_VAL)):
        print("CV2 output is all AIR_HU_VAL! (BUG)")
    else:
        print("CV2 output has content.")

    # Check shapes
    print(f"Scipy shape: {scipy_out.shape}")
    print(f"CV2 shape: {cv2_out.shape}")

if __name__ == "__main__":
    test_rotation()
