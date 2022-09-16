from preprocess_image import images_to_keras_dataset
from deepfake_model import DeepfakeModel

if __name__ == "__main__":
    # Images to keras object
    train_set, validation_set = images_to_keras_dataset()
    # Deepfake model class
    df_mod = DeepfakeModel()
    # Create the model
    model = df_mod.create_model()
    # Train the model
    df_mod.train_model(model, train_set, validation_set)
    # Save the model
    save_model(model, "model")
    
    
