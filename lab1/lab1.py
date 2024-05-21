import cv2

# Read the original image
original_image = cv2.imread('MyPic.png')
# original_image = cv2.imread('A.jpg')

# Get the dimensions of the original image
height, width, channels = original_image.shape

# Number of strips for vertical segmentation
num_strips_vertical = 50

# Calculate the height of each section
section_height = height // num_strips_vertical

# Define variables for starting and ending positions of each section
start_y = 0

# Initialize lists to store even and odd strips
even_strips = []
odd_strips = []

# Loop through each section vertically
for i in range(num_strips_vertical):
    end_y = start_y + section_height  # Calculate the end position of the current section
    
    # Slice the image to extract the current section
    current_section = original_image[start_y:end_y, :]
    
    # Separate even and odd strips
    if i % 2 == 0:
        even_strips.append(current_section)
    else:
        odd_strips.append(current_section)
    
    # Update the starting position for the next section
    start_y = end_y

# Merge even strips vertically
merged_even_vertical = cv2.vconcat(even_strips)

# Merge odd strips vertically
merged_odd_vertical = cv2.vconcat(odd_strips)

# Merge even and odd strips horizontally
merged_image_horizontal = cv2.hconcat([merged_even_vertical, merged_odd_vertical])

# Number of strips for horizontal segmentation
num_strips_horizontal = 100

# Calculate the width of each section
section_width = width // num_strips_horizontal

# Define variables for starting and ending positions of each section
start_x = 0

# Initialize lists to store even and odd strips
even_strips_horizontal = []
odd_strips_horizontal = []

# Loop through each section horizontally
for i in range(num_strips_horizontal):
    end_x = start_x + section_width  # Calculate the end position of the current section
    
    # Slice the image to extract the current section
    current_section = merged_image_horizontal[:, start_x:end_x]
    
    # Separate even and odd strips
    if i % 2 == 0:
        even_strips_horizontal.append(current_section)
    else:
        odd_strips_horizontal.append(current_section)
    
    # Update the starting position for the next section
    start_x = end_x

# Merge even strips horizontally
merged_even_horizontal = cv2.hconcat(even_strips_horizontal)

# Merge odd strips horizontally
merged_odd_horizontal = cv2.hconcat(odd_strips_horizontal)

# Merge even and odd strips vertically
merged_image = cv2.vconcat([merged_even_horizontal, merged_odd_horizontal])

# Save the merged image
cv2.imwrite('final_merged_image.jpg', merged_image)

# Convert the merged image to grayscale
gray_image = cv2.cvtColor(merged_image, cv2.COLOR_BGR2GRAY)

# Save the grayscale image
cv2.imwrite('final_merged_image_gray.jpg', gray_image)

# Display or save the grayscale image as needed
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
