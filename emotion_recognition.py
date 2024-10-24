import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def residual_block(x, filters, kernel_size=3, downsample=False):
    stride = 2 if downsample else 1
    x_shortcut = x

    x = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    if downsample:
        x_shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(x_shortcut)
        x_shortcut = BatchNormalization()(x_shortcut)

    x = tf.keras.layers.add([x, x_shortcut])
    x = Activation('relu')(x)

    return x

if __name__ == "__main__":
    n = 5

    # Real-time preprocessing of the image data
    datagen = ImageDataGenerator(rescale=1.0/255.0, featurewise_center=True, featurewise_std_normalization=True, horizontal_flip=True)

    # Paths to the dataset directories
    train_dir = r'Resnet-Emotion-Recognition\images\train'
    validation_dir = r'Resnet-Emotion-Recognition\images\validation'

    train_generator = datagen.flow_from_directory(
        train_dir,  # Path to the train directory
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=128,
        class_mode='categorical'
    )

    validation_generator = datagen.flow_from_directory(
        validation_dir,  # Path to the validation directory
        target_size=(48, 48),
        color_mode='grayscale',
        batch_size=128,
        class_mode='categorical'
    )

    # Building Residual Network
    inputs = tf.keras.Input(shape=(48, 48, 1))
    x = Conv2D(16, kernel_size=3, padding='same', kernel_regularizer=tf.keras.regularizers.L2(0.0001))(inputs)
    x = residual_block(x, 16, downsample=False)
    for _ in range(n - 1):
        x = residual_block(x, 16)
    x = residual_block(x, 32, downsample=True)
    for _ in range(n - 1):
        x = residual_block(x, 32)
    x = residual_block(x, 64, downsample=True)
    for _ in range(n - 1):
        x = residual_block(x, 64)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # Regression
    outputs = Dense(7, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, decay=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # Fit the model
    model.fit(train_generator, epochs=150, validation_data=validation_generator)

    # Save the trained model
    model.save('current_model/model_resnet_emotion.h5')

    # Evaluate the model
    score = model.evaluate(validation_generator)
    print('Test accuracy:', score)

    # Load and predict (uncomment if needed)
    # model = load_model('current_model/model_resnet_emotion.h5')
    # prediction = model.predict(predict_value)
    # print(prediction)
