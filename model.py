from keras import Model
from keras.callbacks import TensorBoard
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import xception

from preprocessing import *


# Adds new top to base model
def add_top(base):
    x = base.output

    # Global averaging pool layer
    x = GlobalAveragePooling2D()(x)

    # Regular densely connected layer
    x = Dense(512, activation='relu')(x)

    # Output layer
    predictions = Dense(nb_classes, activation='softmax')(x)

    return Model(input=base.input, output=predictions)


# Sets up model for transfer learning
def setup_model(model, base):
    # Freeze the un-trainable layers of the model base
    for layer in base.layers:
        layer.trainable = False

    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )


# Import the Xception model to use as the base for our model
xception_base = xception.Xception(
    include_top=False,
    weights='imagenet',
    input_shape=(img_width, img_height, 3)
)

model = add_top(xception_base)
setup_model(model, xception_base)

# Create TensorBoard logs
tensorboard = TensorBoard(log_dir="logs/transfer-xception", histogram_freq=0, write_graph=True)

# Train the model with data from our generators
model.fit_generator(
    train_generator,
    steps_per_epoch=training_samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=testing_samples // batch_size,
    verbose=1,
    callbacks=[tensorboard]
)

# Save the model to disk
model.save('xception_model.h5')
print("Model saved.")

error_rate = model.evaluate_generator(test_generator)
print("The model's loss rate is {0:0.2} (categorical_crossentropy) and accuracy is {0:.2%}"
      .format(error_rate[0], error_rate[1]))
