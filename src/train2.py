import argparse
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

def build_model(num_classes, input_shape=(224,224,3)):
    base_model = MobileNet(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    return base_model, model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='Directory containing train/val folders')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_out', required=True, help='Path to save trained model')
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

    # Image data generators with augmentation + preprocess
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

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

    base_model, model = build_model(num_classes)

    # ----------------------------------------------------
    # Phase 1 — Train only the classifier head
    # ----------------------------------------------------
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer=Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n=== Training top classifier layers (frozen MobileNet) ===\n")
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=5,
        verbose=1
    )

    # ----------------------------------------------------
    # Phase 2 — Fine-tune the full model
    # ----------------------------------------------------
    print("\n=== Fine-tuning MobileNet + classifier ===\n")

    for layer in base_model.layers:
        layer.trainable = True

    model.compile(
        optimizer=Adam(1e-5),  # smaller LR for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        verbose=1
    )

    # Save final model
    model.save(args.model_out)
    print(f"Model saved to {args.model_out}")

if __name__ == "__main__":
    main()
