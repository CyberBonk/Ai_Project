import cv2
 # pip install opencv-python

def cv2Tester():
  
   
    try:
        # Load the screenshot using OpenCV
        image = cv2.imread('mymap.png')

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Threshold the image to create a binary mask
        ret, mask = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)

        # Invert the mask (black becomes white and vice versa)
        mask = cv2.bitwise_not(mask)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate the area of each contour (assuming they represent roof areas)
        total_area = 0
        for contour in contours:
            total_area += cv2.contourArea(contour)

        print("Total roof area:", total_area, "square pixels")

    except Exception as e:
        print(f"An error occurred: {e}")
