import argparse
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

def cross_correlation(img, kernel):
    
    #get kernel width and height 
    kernel_height,kernel_width = kernel.shape
    #initializes a new image has the same shape as the input image but filled with zeros. 
    #stores the result of the cross-correlation operation.
    new_img = np.zeros(img.shape)

    #calculate the amount of padding needed for each side of the image
    kernel_height_pad = (kernel_height - 1) // 2
    kernel_width_pad = (kernel_width - 1) // 2

    x,y = img.shape

    # created padded img
    padded_img = np.pad(img, pad_width=((kernel_height_pad, kernel_height_pad,), (kernel_width_pad, kernel_width_pad)), mode = 'constant', constant_values = 0)

        #loop 
    for i in range(x):
            for j in range(y):
                window = padded_img[i:i + kernel_height, j:j + kernel_width]  #extract a region of interest (ROI) from the padded image
                if window.shape == kernel.shape:
                        new_img[i, j] = np.sum(kernel * window) #perform cross-correlation operation with the given kernel


    return new_img

def convolution(img, kernel):

    # flip kernel
    flipped_kernel = np.flip(kernel)

    # convulution by cross-correlation function with a flipped kernel
    conv_image = cross_correlation(img, flipped_kernel)

    return conv_image

def gaussian_blur(sigma, height, width):

    #Creating the Gaussian Kernel:
    gauss_kernel = np.zeros((height, width))
    h = np.linspace(-height / 2 + 1, height / 2, height)
    w = np.linspace(-width / 2 + 1, width / 2, width)

    #Filling the Gaussian Kernel
    for x, x1 in enumerate(h):
        for y, y1 in enumerate(w):
            #calculates the value of the Gaussian kernel at position (x, y) using the formula for a 2D Gaussian distribution
            gauss_kernel[x,y] = 1 / (2 * np.pi * (sigma ** 2)) * np.exp(-(x1 ** 2 + y1 ** 2)/(2 * (sigma ** 2)))
    
    #Normalizing the Gaussian Kernel
    norm_gauss = gauss_kernel * 1 / np.sum(gauss_kernel)
    
    return norm_gauss


def low_pass(img, sigma, size):
    
    return convolution(img, gaussian_blur(sigma, size, size))

def high_pass(img, sigma, size):
    
    return img - low_pass(img, sigma, size)

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    # Resize img2 to match the dimensions of img1
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)


def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        return img
    return None

def main():
    # Select input images
    print("Select the first image:")
    img1 = open_image()
    if img1 is None:
        print("No image selected. Exiting.")
        return
    print("Select the second image:")
    img2 = open_image()
    if img2 is None:
        print("No image selected. Exiting.")
        return

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Define parameters for creating hybrid image
    sigma1 = 10  
    size1 = 30   
    high_low1 = 'low'
    sigma2 = 10   
    size2 = 25   
    high_low2 = 'high'
    mixin_ratio = 0.5  
    scale_factor = 1.2  

    # Create hybrid image
    hybrid_img = create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2, high_low2, mixin_ratio, scale_factor)

    # Display or save the resulting hybrid image
    cv2.imshow('Hybrid Image', hybrid_img)
    cv2.imwrite('suan_lab02_hybrid.png', hybrid_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()