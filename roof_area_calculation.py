from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from geopy.geocoders import Nominatim
import folium
import os
import cv2

def main():
    # Initialize Nominatim geocoder
    geolocator = Nominatim(user_agent="my_geocoder")

    # Define the address
    address = "1600 Pennsylvania Avenue NW, Washington, D.C."

    # Get the coordinates
    location = geolocator.geocode(address)

    if location:
        print("Latitude:", location.latitude)
        print("Longitude:", location.longitude)
    else:
        print("Geocoding failed. Check the address or try a different one.")
        return

    # Create a map centered at a specific location
    mymap = folium.Map(location=[location.latitude, location.longitude], zoom_start=18.5)

    # Add a satellite imagery layer
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite Imagery',
        overlay=True,
        control=True,
    ).add_to(mymap)

    # Save the map as an HTML file
    mymap.save('mymap.html')

    # Set up the Chrome driver
    driver = None
    try:
        # Initialize ChromeOptions
        chrome_options = webdriver.ChromeOptions()
        # Uncomment the next line if you want to run the browser in headless mode
        # chrome_options.add_argument("--headless")

        # Use ChromeDriverManager to manage the driver
        service = Service(ChromeDriverManager().install())

        # Set up the driver with the service and options
        driver = webdriver.Chrome(service=service, options=chrome_options)
        print("WebDriver is running successfully!")

        # Open the saved map file
        driver.get(f"file://{os.path.abspath('mymap.html')}")

        # Take a screenshot
        driver.save_screenshot('mymap.png')

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
    finally:
        # Close the browser (cleanup)
        if driver:
            driver.quit()

if __name__ == "__main__":
    main()
