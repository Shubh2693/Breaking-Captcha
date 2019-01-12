import os
import os.path
import cv2
import glob
import imutils

import numpy as np
import pandas as pd
import sklearn
from joblib import load

from collections import Counter

############################################################################################################
#
# Feature Extraction
#
#############################################################################################################
# flag = "train"
flag = "test"

if(flag == "train"):
    CAPTCHA_IMAGE_FOLDER = "generated_captcha_image"
    OUTPUT_FOLDER = "extracted_letter_images"
elif flag == "test":
    CAPTCHA_IMAGE_FOLDER = "black_test_captcha"
    OUTPUT_FOLDER = "test_extracted_letter_images"

# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}

kernel = np.ones((5,2),np.uint8) # For Erosion

total_captchas = 0
right_predicted = 0
partial_prediction_array = [0,0,0,0,0]

cc=0
# loop over the image paths
for (i, captcha_image_file) in enumerate(captcha_image_files):

    # Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
    # grab the base filename as the text
    filename = os.path.basename(captcha_image_file)
    captcha_correct_text = os.path.splitext(filename)[0]
    

    # Load the image and convert it to grayscale
    image = cv2.imread(captcha_image_file)

    

    ## Erosion ###
    # Dilation followed by erosion
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # Erosion followed by dilation
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    gray = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
    
    # threshold the image (convert it to pure black and white)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # find the contours (continuous blobs of pixels) the image
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Hack for compatibility with different OpenCV versions
    contours = contours[0] if imutils.is_cv2() else contours[1]

    letter_image_regions = []

    clf = load('classifier.joblib')
    # Now we can loop through each of the four contours and extract the letter
    # inside of each one
    for contour in contours:
        # Get the rectangle that contains the contour
        (x, y, w, h) = cv2.boundingRect(contour)

        if w < 12 or h < 10:
            continue

        # Compare the width and height of the contour to detect letters that
        # are conjoined into one chunk
        if w > 40:
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
            half_width = int(w / 2)
            letter_image_regions.append((x, y, half_width, h))
            letter_image_regions.append((x + half_width, y, half_width, h))
        else:
            # This is a normal letter by itself
            letter_image_regions.append((x, y, w, h))


    # Sort the detected letter images based on the x coordinate to make sure
    # we are processing them from left-to-right so we match the right image
    # with the right letter
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

    if(len(letter_image_regions) != 4):
        continue

    predicted_CAPTCHA = ""

    frequency_dict = Counter(captcha_correct_text)

    # Save out each letter as a single image
    for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box

        # Extract the letter from the original image with a 2-pixel margin around the edge
        if h >= 60 or y == 0 or y==1:
            letter_image = gray[y:y + h, x - 2:x + w + 2]
        else:
            letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]
        
        if letter_image.any():
            letter_image = cv2.resize(letter_image, (50,60), interpolation=cv2.INTER_AREA)
        else:
            # print("Empty")
            continue

        data_pixels = []

        for x in range(0,50):
            for y in range(0,60):
                data_pixels.append(letter_image[y][x])

        data_vector = [str(i) for i in data_pixels]
        data_vector2 = np.reshape(data_vector, (1,3000))

        ############################################################################################################
        #
        # Prediction
        #
        #############################################################################################################

        predicted_letter = clf.predict(data_vector2)
        
        predicted_CAPTCHA = predicted_CAPTCHA + predicted_letter

        # print(len(data_pixels))


        # Get the folder to save the image in
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)

        # # if the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # # write the letter image to a file
        count = counts.get(letter_text, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)

        # increment the count for the current key
        counts[letter_text] = count + 1
        if str(predicted_letter) in frequency_dict:
            frequency_dict[str(predicted_letter)] -= 1
            if frequency_dict[str(predicted_letter)] >= 0:
                partial_prediction += 1
    cc+=1
    print("*************************************************")

    print("CAPTCHA Text = ",captcha_correct_text)
    total_captchas += 1

    # partial_prediction_array[partial_prediction] += 1

    print("Prediction: ",predicted_CAPTCHA)
    if predicted_CAPTCHA == captcha_correct_text:
        right_predicted += 1
  
print("Right Predicted: ", right_predicted, "out of ", total_captchas )
print(partial_prediction_array)

