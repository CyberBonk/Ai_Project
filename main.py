#from IPython.display import Image

import folium
import time

from geopy.geocoders import Nominatim

import asyncio
from pyppeteer import launch

from selenium import webdriver
import os


# Initialize Nominatim geocoder
geolocator = Nominatim(user_agent="my_geocoder")

# Define the address
address = "1600 Pennsylvania Avenue NW, Washington, D.C."

# Get the coordinates
location = geolocator.geocode(address)

# Print latitude and longitude   test
print("Latitude:", location.latitude)
print("Longitude:", location.longitude)

#Image(r"C:\Users\Bebo\Desktop\Artificial_Intelligence\Project\testing.png").save("output.png")

# Create a map centered at a specific location
mymap = folium.Map(location=[location.latitude ,location.longitude], zoom_start=100)

# Add a satellite imagery layer (you can replace the URL with other sources)
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Satellite Imagery',
    overlay=True,
    control=True,
).add_to(mymap)

# Save the map as a PNG file
mymap.save('mymap.html')



# Set up the Firefox driver (you can use other browsers too)

try:
    # Get the current directory where the script is located
    current_directory = os.path.dirname(os.path.abspath(__file__))
    # Specify the relative path to mymap.html
    relative_path = os.path.join(current_directory, 'mymap.html')
    driver = webdriver.Firefox()
    driver.get('file://' + relative_path)  # Use 'file://' prefix for local file paths
    print("WebDriver is running successfully!")
except Exception as e:
    print(f"Error: {e}")

time.sleep(2)
# Capture the screenshot and save it as mymap.png
driver.save_screenshot('mymap.png')

# Close the browser window
driver.quit()




#pypeteer as a backup for follium, but not needed since firefox gecko is working fine
'''async def main():
    browser = await launch()
    page = await browser.newPage()

    relative_path = r'mymap.html'   #possible point to look onto, how to get the local driectory of mymap.html

    #await page.goto('file:///path/to/mymap.html')
    await page.goto(relative_path)
    
    await page.screenshot({'path': 'mymap.png'})
    await browser.close()

asyncio.get_event_loop().run_until_complete(main())
'''