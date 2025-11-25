import argparse
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

def build_model(num_classes, input_shape=(224,224,3), learning_rate=1e-4):
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Directory containing train/val folders')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_out', required=True, help='Path to save trained model')
    args = parser.parse_args()

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    # Image data generators
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        args.data_dir,
        target_size=(224,224),
        batch_size=args.batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        args.data_dir,
        target_size=(224,224),
        batch_size=args.batch_size,
        class_mode='categorical',
        subset='validation'
    )

    num_classes = len(train_generator.class_indices)
    print(f"Detected {num_classes} classes: {train_generator.class_indices}")

    model = build_model(num_classes)

    model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=val_generator,
        verbose=1
    )

    model.save(args.model_out)
    print(f"Model saved to {args.model_out}")

if __name__ == "__main__":
    main()
