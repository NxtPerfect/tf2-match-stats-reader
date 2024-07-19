import cv2
import pytesseract

def loadImage(PATH):
    return cv2.imread(PATH)

def findContours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Performing OTSU threshold
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
                                                 cv2.CHAIN_APPROX_NONE)
    return contours

def extractTextFromImage(contours, image):
    # return pytesseract.image_to_string(image)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
    
        # Drawing a rectangle on copied image
        rect = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Cropping the text block for giving input to OCR
        cropped = image[y:y + h, x:x + w]
        
        # Open the file in append mode
        file = open("output/recognized.txt", "a")
        
        # Apply OCR on the cropped image
        text = pytesseract.image_to_string(cropped)
        
        # Appending the text into file
        file.write(text)
        file.write("\n")
        
        # Close the file
        file.close()
        print(f"Found text {text}")

def run():
    image = loadImage("./data/TF2TestLowContrast.jpg")
    contours = findContours(image)
    extractTextFromImage(contours, image.copy())

    with open("./output/recognized.txt") as f:
        print(f.read())

if __name__ == "__main__":
    run()
