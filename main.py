#from IPython.display import Image

import folium

from geopy.geocoders import Nominatim

import asyncio
from pyppeteer import launch

#from selenium import webdriver



# Initialize Nominatim geocoder
geolocator = Nominatim(user_agent="my_geocoder")

# Define the address
address = "1600 Pennsylvania Avenue NW, Washington, D.C."

# Get the coordinates
location = geolocator.geocode(address)

# Print latitude and longitude
print("Latitude:", location.latitude)
print("Longitude:", location.longitude)

#Image(r"C:\Users\Bebo\Desktop\Artificial_Intelligence\Project\testing.png").save("output.png")

# Create a map centered at a specific location
mymap = folium.Map(location=[location.latitude ,location.longitude], zoom_start=1000)

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


'''    SELENIUM IS NOT RUNNING RIGHT NOW, TRYING ALTERNATIVES

# Set up the Chrome driver (you can use other browsers too)
try:
    # Initialize the Chrome WebDriver
    driver = webdriver.Chrome()
    print("WebDriver is running successfully!")
except Exception as e:
    print(f"Error: {e}")

# Navigate to your HTML file (replace 'file:///path/to/mymap.html' with the actual file path)
relative_path = r'mymap.html'
driver.get(relative_path)
# Capture the screenshot and save it as mymap.png
driver.save_screenshot('mymap.png')

# Close the browser window
driver.quit()

'''
async def main():
    browser = await launch()
    page = await browser.newPage()

    relative_path = r'mymap.html'   #possible point to look onto, how to get the local driectory of mymap.html

    #await page.goto('file:///path/to/mymap.html')
    await page.goto(relative_path)
    
    await page.screenshot({'path': 'mymap.png'})
    await browser.close()

asyncio.get_event_loop().run_until_complete(main())