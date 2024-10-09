import tensorflow as tf

image_shape = (100, 100)
input_shape = (100, 100, 3)
fruits_quantity = 131
batch_size = 32
epochs = 5
learning_rate = 0.0001
train_dir = "fruits-360/Training"
test_dir = "fruits-360/Test"

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

base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(units=fruits_quantity, activation='softmax')

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
inputs = tf.keras.Input(shape=input_shape)
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(train_data, epochs=epochs, validation_data=val_data)
model.evaluate(test_data)
