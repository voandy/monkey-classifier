import sys

import numpy as np
from keras import optimizers
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

from preprocessing import *

# Squelch TensorFlow debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

monkey_labels = {
    0: "Mantled Howler",
    1: "Patas Monkey",
    2: "Bald Uakari",
    3: "Japanese Macaque",
    4: "Pygmy Marmoset",
    5: "White-headed Capuchin",
    6: "Silvery Marmoset",
    7: "Common Squirrel Monkey",
    8: "Black-headed Night Monkey",
    9: "Nilgiri Langur",
}

# Load model
model = load_model('xception_model_ft.h5')
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
    metrics=['accuracy']
)


# Prints a prediction
def print_prediction(best_guess, prob):
    print("Species: " + monkey_labels[best_guess])
    print("Probability: {0:.2%}\n".format(prob))


# Loads an image and makes a prediction using the model
def predict_image(filename):
    # Load image, downsize, scale and convert to array
    test_image = img_to_array(load_img(filename, target_size=(img_width, img_height))) / 255.0

    # Expand array by 1 to match model
    test_image = np.expand_dims(test_image, axis=0)

    # Calculate category probabilities using model
    predictions = model.predict(test_image)[0]

    # Get label of highest probability prediction
    best_guess = 0
    highest_prob = 0.0
    for i in range(len(predictions)):
        if predictions[i] > highest_prob:
            best_guess = i
            highest_prob = predictions[i]

    print_prediction(best_guess, highest_prob)


if len(sys.argv) > 1:
    predict_image(sys.argv[1])
