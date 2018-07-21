from keras import optimizers
from keras.callbacks import TensorBoard

from preprocessing import *
from keras.models import load_model


def setup_model(model, trainable):
    # Freeze the un-trainable layers of the model base
    for layer in model.layers[:(len(model.layers) - trainable)]:
        layer.trainable = False

    for layer in model.layers[(len(model.layers) - trainable):]:
        layer.trainable = True

    model.compile(
        loss='categorical_crossentropy',
        # Slower training rate for fine-tuning
        optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
        metrics=['accuracy']
    )


# Load trained model
model = load_model('xception_model.h5')

# Setup model to retrain our top layers plus block 13 and 14 of Xception
setup_model(model, 19)

# Create TensorBoard logs
tensorboard = TensorBoard(log_dir="logs/ft-exception", histogram_freq=0, write_graph=True)

# model.summary()
print(model.evaluate_generator(test_generator))

# Train the model with data from our generators
model.fit_generator(
    train_generator,
    steps_per_epoch=training_samples // batch_size,
    epochs=epochs_ft,
    validation_data=test_generator,
    validation_steps=testing_samples // batch_size,
    verbose=1,
    callbacks=[tensorboard]
)

# Save the model to disk
model.save('xception_model_ft.h5')
print("Model saved.")

error_rate = model.evaluate_generator(test_generator)
print("The model's loss rate is {0:0.2} (categorical_crossentropy) and accuracy is {0:.2%}"
      .format(error_rate[0], error_rate[1]))
