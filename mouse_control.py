import pyautogui
import math
import time

# Function to move the mouse slowly in a circular motion
def move_mouse_slowly(duration, radius, cx, cy):
    for angle in range(0, 360, 5):
        x = cx + radius * math.cos(math.radians(angle))
        y = cy + radius * math.sin(math.radians(angle))
        pyautogui.moveTo(x, y, duration=0.1)

# Function to click the left mouse button at the center of the screen
def click_center():
    pyautogui.click(screen_width // 2, screen_height // 2)

# Get the current screen width and height
screen_width, screen_height = pyautogui.size()

# Set the center of the circular motion to the middle of the screen
center_x = screen_width // 2
center_y = screen_height // 2

# Set the radius of the circular motion
radius = 100

start_time = time.time()

# Run the script indefinitely
while True:
    move_mouse_slowly(1, radius, center_x, center_y)
    
    if time.time() - start_time >= 4:
        click_center()
        start_time = time.time()
