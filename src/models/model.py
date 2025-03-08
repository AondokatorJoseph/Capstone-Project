import tensorflow as tf
from tensorflow.keras import layers, models

class WasteClassifier:
    def __init__(self, num_classes=6, input_shape=(224, 224, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None

    def build_model(self):
        """Build the CNN model architecture"""
        base_model = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape
        )
        
        # Freeze the base model layers
        base_model.trainable = False

        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model

    def train(self, train_data, validation_data, epochs=10, batch_size=32):
        """Train the model"""
        if self.model is None:
            self.build_model()
            
        history = self.model.fit(
            train_data[0], train_data[1],
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size
        )
        
        return history