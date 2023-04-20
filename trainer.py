from tensorflow.keras.optimizers import Adam
from data_loader import get_train_valid_data
from models import inception_v1

# Load the data
train_data, valid_data, num_classes = get_train_valid_data()

# Load the Inception V1 model
model = inception_v1(input_shape=(224, 224, 3), num_classes=num_classes)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
epochs = 10
history = model.fit(train_data,
                    epochs=epochs,
                    validation_data=valid_data)
