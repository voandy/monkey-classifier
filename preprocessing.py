import os.path
from keras.preprocessing.image import ImageDataGenerator

img_width = 197
img_height = 197

epochs = 100
epochs_ft = 200
batch_size = 120

train_path = 'training-data/training'
validation_path = 'training-data/validation'

nb_classes = 10

# Define a seed to remove some randomness for testing purposes
# seed = 24601

# Counts the number of training and testing samples in the directories
training_samples = sum([len(files) for r, d, files in os.walk(train_path)])
testing_samples = sum([len(files) for r, d, files in os.walk(validation_path)])

# Augment images to prevent over-fitting and help the model identify true features
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.2,
    rescale=1.0 / 255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,
)

# Test image data only need to be rescaled to floats between 0 and 1
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Defines pipelines for training and testing data that loads batches of images, converts them to 3D numpy arrays
# and returns an iterator yielding the batches and their labels
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    # seed=seed,
    class_mode='categorical',
)

test_generator = validation_datagen.flow_from_directory(
    validation_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    # seed=seed,
    class_mode='categorical',
)
