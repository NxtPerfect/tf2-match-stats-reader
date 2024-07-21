import cv2
import pytesseract
import numpy as np
from os import listdir, makedirs
from os.path import join, basename, splitext, exists

def loadImage(PATH):
    return cv2.imread(PATH)

def preprocess_image(image):
    # Get the bottom 1/8th of the image and trim 1/8th from each side
    height, width = image.shape[:2]
    bottom_section = image[int(7*height/8):height, int(width/8):int(7*width/8)]
    
    # Convert to grayscale
    gray = cv2.cvtColor(bottom_section, cv2.COLOR_BGR2GRAY)
    
    # Apply very mild Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply mild sharpening
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)
    
    # Increase contrast slightly
    alpha = 1.2  # Contrast control (1.0-3.0)
    beta = 0  # Brightness control (0-100)
    adjusted = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)
    
    return adjusted

def findContours(image):
    # Apply edge detection
    edges = cv2.Canny(image, 100, 200)
    
    # Dilate the edges to connect nearby text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def extractTextFromImage(contours, image):
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter out very small contours
        if w < 20 or h < 10:
            continue
        
        # Cropping the text block for giving input to OCR
        cropped = image[y:y + h, x:x + w]
        
        # Apply OCR on the cropped image
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789: '
        text = pytesseract.image_to_string(cropped, config=custom_config).strip()
        
        if text:
            print(f"Found text: {text}")
            with open("output/recognized.txt", "a") as file:
                file.write(text + "\n")

def run():
    # Create output directory if it doesn't exist
    output_dir = "output"
    if not exists(output_dir):
        makedirs(output_dir)

    images = [f"./data/{x}" for x in listdir("./data/") if x.endswith((".jpg", ".png"))]
    for path in images:
        print(f"Processing {path}")
        image = loadImage(path)
        preprocessed = preprocess_image(image)
        
        # Save preprocessed image
        filename = splitext(basename(path))[0]
        output_path = join(output_dir, f"{filename}_preprocessed.png")
        cv2.imwrite(output_path, preprocessed)
        print(f"Saved preprocessed image to {output_path}")
        
        contours = findContours(preprocessed)
        extractTextFromImage(contours, preprocessed)

    with open("./output/recognized.txt") as f:
        print("Recognized text:")
        print(f.read())

if __name__ == "__main__":
    run()
