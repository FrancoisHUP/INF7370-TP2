import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt
import zipfile
import os
from tqdm import tqdm
from pathlib import Path
import random
import matplotlib.pyplot as plt
from pathlib import Path

class DataLoader:
    def __init__(self, data_path="donnees/", 
                 train_subdir="entrainement", 
                 test_subdir="test",
                 images_color_mode='grayscale', 
                 image_scale=224, 
                 training_batch_size=32, 
                 validation_batch_size=8):
        """
        Initializes the DataLoader.

        Args:
            data_path (str): Path to a ZIP file or an already extracted dataset directory.
            train_subdir (str): Subdirectory for training data.
            test_subdir (str): Subdirectory for test data.
            images_color_mode (str): Color mode for images ('rgb', 'grayscale', etc.).
            image_scale (int): Target dimension for images (both width and height).
            training_batch_size (int): Batch size for training.
            validation_batch_size (int): Batch size for validation.
        """
        self.train_subdir = train_subdir
        self.test_subdir = test_subdir
        self.data_path = Path(data_path)
        self.image_scale = image_scale        

        # Determine the number of channels based on color mode
        if images_color_mode.lower() == 'rgb':
            self.image_channels = 3
        else:
            self.image_channels = 1
        
        # If data_path is a zip file, then extract it if needed.
        if self.data_path.is_file() and self.data_path.suffix == '.zip':
            print(f"Extracting '{self.data_path}' to '.' ...")
            self.extract_data(self.data_path, '.')
            self.data_path = Path('.')
        
        if not self.data_path.is_dir():
            raise ValueError(f"{self.data_path} is not a valid directory.")
        
        # Build full paths for training and validation directories.
        train_path = self.data_path / train_subdir
        
        # Initialize the data generators.
        self.training_data_generator = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        self.training_generator = self.training_data_generator.flow_from_directory(
            str(train_path),
            color_mode=images_color_mode,
            target_size=(image_scale, image_scale),
            batch_size=training_batch_size,
            class_mode="categorical",
            shuffle=True,
            subset="training"
        )
        
        self.validation_generator = self.training_data_generator.flow_from_directory(
            str(train_path),
            color_mode=images_color_mode,
            target_size=(image_scale, image_scale),
            batch_size=validation_batch_size,
            class_mode="categorical",
            shuffle=True,
            subset="validation"
        )

    def extract_data(self, zip_path: Path, extraction_dir: Path):
        """
        Extracts a ZIP file into the given directory with a progress bar.
        
        Args:
            zip_path (Path): Path to the zip file.
            extraction_dir (Path): Directory where the contents will be extracted.
        """        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file in tqdm(zip_ref.namelist(), desc="Extracting files"):
                zip_ref.extract(member=file, path=extraction_dir)

    def load_batch(self):
        """
        Loads a batch of training and validation data.
        
        Returns:
            Tuple containing (x_train, y_train, x_val, y_val)
        """
        x_train, y_train = next(self.training_generator)
        x_val, y_val = next(self.validation_generator)
        return x_train, y_train, x_val, y_val

    def display_random_sample(self):
        """
        Displays one random image from each class in the training directory in a single plot.
        """
        train_path = self.data_path / self.train_subdir
        class_images = []
        class_names = []
        
        # Iterate over each class folder in the training directory
        for class_dir in train_path.iterdir():
            if class_dir.is_dir():
                images = list(class_dir.glob('*'))
                if images:
                    # Pick one random image for this class
                    selected_img = random.choice(images)
                    class_images.append(selected_img)
                    class_names.append(class_dir.name)
        
        # Determine the number of classes and create a subplot accordingly.
        num_classes = len(class_images)
        fig, axes = plt.subplots(1, num_classes, figsize=(num_classes * 3, 3))
        
        # In case there's only one class, wrap axes in a list
        if num_classes == 1:
            axes = [axes]
            
        for ax, img_path, class_name in zip(axes, class_images, class_names):
            img = plt.imread(str(img_path))
            ax.imshow(img)
            ax.set_title(class_name)
            ax.axis("off")
        
        plt.tight_layout()
        plt.show()

class ModelHandler: 
    def __init__(self, image_scale=224, image_channels=1):
        self.image_shape = (image_scale, image_scale, image_channels)
        self.model = self.model_structure()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # loss="categorical_crossentropy" of "mse" 

        os.makedirs("models/", exist_ok=True)
        os.makedirs("output/", exist_ok=True)

    def model_structure(self):
        # Build the model using the Sequential API
        model = Sequential([
            Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=self.image_shape),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(64, (3, 3), padding='same', activation='relu'),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(128, (3, 3), padding='same', activation='relu'),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(256, (3, 3), padding='same', activation='relu'),
            MaxPooling2D((2, 2), padding='same'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(6, activation='softmax')
        ])
        return model
    
    # def _forward_pass(self, input_data): 
    #     return self.model(input_data)
    
    def train(self, data_loader, epochs=10, modelsPath="Model.keras"):
        # Create a TensorBoard callback that logs metrics every 10 batches
        tensorboard_callback = TensorBoard(log_dir="logs/", update_freq=10)
        
        # Create the checkpoint callback as before
        modelcheckpoint = ModelCheckpoint(filepath="models/Model.keras",
                                        monitor='val_accuracy',
                                        verbose=1,
                                        save_best_only=True,
                                        mode='auto')
        
        # Then, when calling fit:
        classifier = self.model.fit(
            data_loader.training_generator,
            epochs=epochs,
            validation_data=data_loader.validation_generator,
            callbacks=[modelcheckpoint, tensorboard_callback],
            shuffle=True
        )
        return classifier

    def plot_training_history(self, classifier, output_path="output/accuracy_plot.png"): 
        # Check for different key names (depending on your Keras version)
        history = classifier.history
        # Try to detect which keys are present:
        if 'accuracy' in history:
            acc_key = 'accuracy'
            val_acc_key = 'val_accuracy'
        elif 'acc' in history:
            acc_key = 'acc'
            val_acc_key = 'val_acc'
        else:
            print("Accuracy keys not found in history.")
            return

        # Create the figure
        plt.figure()
        plt.plot(history[acc_key], label='Train Accuracy')
        plt.plot(history[val_acc_key], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        
        # Save the figure to a file
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
        
        plt.show()

def main():
    # Initialize the DataLoader (this will handle zip extraction or directory usage automatically)
    data_loader = DataLoader("donnees/", images_color_mode='grayscale')  # or "donnees.zip" if you have the zip file

    # Display random samples from each class
    # loader.display_random_sample()

    model_handler = ModelHandler(image_scale=data_loader.image_scale,
                                 image_channels=data_loader.image_channels)
    model_handler.model.summary()

    classifier = model_handler.train(data_loader, epochs=10, modelsPath="models/Model.keras")

    model_handler.plot_training_history(classifier)


if __name__ == "__main__":
    main()
