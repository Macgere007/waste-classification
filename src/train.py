import argparse
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def build_cnn(input_shape=(128,128,3), num_classes=2):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2,2)),
        BatchNormalization(),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        BatchNormalization(),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        BatchNormalization(),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Directory containing train/ and val/ folders')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_out', required=True, help='Path to save trained model')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise ValueError("data_dir must contain 'train/' and 'val/' folders")

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(64,64),
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(64,64),
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=False
    )

    num_classes = len(train_generator.class_indices)
    print(f"Detected {num_classes} classes: {train_generator.class_indices}")

    model = build_cnn(input_shape=(64,64,3), num_classes=num_classes)
    model.summary()

    model.fit(
        train_generator,
        epochs=args.epochs,
        validation_data=val_generator,
        verbose=1
    )

    model.save(args.model_out)
    print(f"\nModel saved to {args.model_out}")

if __name__ == "__main__":
    main()
