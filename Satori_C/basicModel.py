import tensorflow as tf

image_shape = (100, 100)
input_shape = (100, 100, 3)
fruits_quantity = 131
batch_size = 32
epochs = 5
train_dir = "fruits-360/Training"
test_dir = "fruits-360/Test"

CNN = tf.keras.Sequential()
CNN.add(tf.keras.layers.Rescaling(1. / 255.0))
CNN.add(tf.keras.layers.RandomFlip('horizontal'))
CNN.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=input_shape))
CNN.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
CNN.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
CNN.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
CNN.add(tf.keras.layers.Flatten())
CNN.add(tf.keras.layers.Dense(units=256, activation='relu'))
CNN.add(tf.keras.layers.Dense(units=fruits_quantity, activation='softmax'))
CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

train_data, val_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset='both',
    label_mode='categorical',
    seed=50,
    image_size=image_shape,
    batch_size=batch_size
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    label_mode='categorical',
    image_size=image_shape,
    batch_size=batch_size
)

CNN.fit(train_data, epochs=epochs, validation_data=val_data)
CNN.evaluate(test_data)
