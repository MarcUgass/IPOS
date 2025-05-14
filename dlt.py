import cv2
import numpy as np
import argparse
import os

def order_points(pts):
    
    if pts.shape != (4, 2):
        print(f"Error: Expected 4 points, but got {pts.shape[0]}")
        return None

    # Initialize a list of coordinates that will be ordered
    rect = np.zeros((4, 2), dtype="float32")

   
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left
    rect[2] = pts[np.argmax(s)] # Bottom-right

   
    diff = np.diff(pts, axis=1) # Computes y - x
    rect[1] = pts[np.argmin(diff)] # Top-right
    rect[3] = pts[np.argmax(diff)] # Bottom-left

    # Return the ordered coordinates
    return rect

def perspective_transform(image, pts):
    
    rect = order_points(pts)
    if rect is None:
        return None
    (tl, tr, br, bl) = rect

    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

   
    dst = np.array([
        [0, 0],                # Top-left
        [maxWidth - 1, 0],     # Top-right
        [maxWidth - 1, maxHeight - 1], # Bottom-right
        [0, maxHeight - 1]     # Bottom-left
    ], dtype="float32")

    
    M = cv2.getPerspectiveTransform(rect, dst)
    if M is None:
        print("Error: Could not compute perspective transform matrix.")
        return None

    
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def find_document_contour(image):
 
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = cv2.resize(image, (int(image.shape[1] / ratio), 500))


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    print("STEP 1: Edge Detection")


    contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None 

    print("STEP 2: Find contours of paper")

    for c in contours:

        peri = cv2.arcLength(c, True) 

        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        print(len(approx))
        if len(approx) == 4:
            screenCnt = approx
            print(f"Found contour with 4 points: {screenCnt.reshape(4, 2)}")
            break 

    if screenCnt is None:
        
        return None


    return screenCnt.reshape(4, 2) * ratio


# --- Main Execution ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the input image")
    ap.add_argument("-o", "--output", required=True, help="Path to save the output warped image")
    args = vars(ap.parse_args())

    input_image_path = args["image"]
    output_image_path = args["output"]

    # --- Input Validation ---
    if not os.path.exists(input_image_path):
        print(f"Error: Input image not found at {input_image_path}")
        exit(1)

    output_dir = os.path.dirname(output_image_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            exit(1)


    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: Could not load image from {input_image_path}")
        exit(1)

    doc_contour_points = find_document_contour(image)

    if doc_contour_points is None:
        print("Failed to find document contour. Exiting.")
        exit(1)

    print("STEP 3: Apply perspective transform")
    warped_image = perspective_transform(image.copy(), doc_contour_points) # Use copy of original

    if warped_image is None:
         print("Failed to apply perspective transform. Exiting.")
         exit(1)

    print(f"STEP 4: Saving warped image to {output_image_path}")
    warped_image = cv2.resize(warped_image, (1192, 1682))
    success = cv2.imwrite(output_image_path, warped_image)
    if not success:
        print(f"Error: Failed to save warped image to {output_image_path}")
        exit(1)
    else:
        print("Warped image saved successfully.")

 
    display_height = 1200
    orig_h, orig_w = image.shape[:2]
    warped_h, warped_w = warped_image.shape[:2]

    orig_display_w = int(orig_w * (display_height / orig_h))
    warped_display_w = int(warped_w * (display_height / warped_h))

    cv2.imshow("Original", cv2.resize(image, (orig_display_w, display_height)))
    cv2.imshow("Warped", cv2.resize(warped_image, (warped_display_w, display_height)))
    print("Press any key to close windows.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

