import os
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout, Input, GlobalAveragePooling2D 
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
import matplotlib.pyplot as plt
import zipfile
from tqdm import tqdm
from pathlib import Path
import re
import random


class DataLoader:
    def __init__(self, config):
        """
        Initializes the DataLoader.

        Reads both data paths and augmentation parameters from config.
        """
        self.data_config = config["data"]
        self.aug_config = config["augmentation"]
        # Use model config to set image size and color mode.
        model_config = config.get("model", None)
        if model_config:
            self.image_scale = model_config["input_shape"][0]
            channels = model_config["input_shape"][-1]
            if channels == 1:
                self.images_color_mode = 'grayscale'
                self.image_channels = 1
            else:
                self.images_color_mode = 'rgb'
                self.image_channels = 3
        else:
            self.image_scale = 224
            self.images_color_mode = 'grayscale'
            self.image_channels = 1

        self.setup_paths(self.data_config["data_path"],
                         self.data_config["train_subdir"],
                         self.data_config["test_subdir"])
        self.extract_data_if_needed()

        # Build full paths for training directory.
        train_path = self.data_path / self.train_subdir

        # Initialize the training data generator using augmentation config.
        self.training_data_generator = ImageDataGenerator(
            rescale=self.aug_config.get("rescale", 1./255),
            rotation_range=self.aug_config.get("rotation_range", 0),
            width_shift_range=self.aug_config.get("width_shift_range", 0),
            height_shift_range=self.aug_config.get("height_shift_range", 0),
            zoom_range=self.aug_config.get("zoom_range", 0),
            brightness_range=self.aug_config.get("brightness_range", None),
            horizontal_flip=self.aug_config.get("horizontal_flip", False),
            validation_split=self.aug_config.get("validation_split", 0.1)
        )

        self.training_generator = self.training_data_generator.flow_from_directory(
            str(train_path),
            color_mode=self.images_color_mode,
            target_size=(self.image_scale, self.image_scale),
            batch_size=self.data_config.get("batch_size_training", 32),
            class_mode="categorical",
            shuffle=True,
            subset="training"
        )

        self.validation_generator = self.training_data_generator.flow_from_directory(
            str(train_path),
            color_mode=self.images_color_mode,
            target_size=(self.image_scale, self.image_scale),
            batch_size=self.data_config.get("batch_size_validation", 8),
            class_mode="categorical",
            shuffle=True,
            subset="validation"
        )

    def setup_paths(self, data_path, train_subdir, test_subdir):
        """Converts the data path to a Path object and sets training/test subdirectories."""
        self.data_path = Path(data_path)
        self.train_subdir = train_subdir
        self.test_subdir = test_subdir

    def extract_data_if_needed(self):
        """
        If the provided data_path is a zip file, extract it into a folder whose name
        is the stem of the zip file (e.g., 'donnees' for 'donnees.zip') and update data_path.
        """
        if self.data_path.is_file() and self.data_path.suffix == '.zip':
            # Set the extraction folder as the stem of the zip file.
            extraction_folder = self.data_path.with_suffix('')
            if extraction_folder.exists():
                print(f"Found extracted folder '{extraction_folder}'.")
            else:
                print(f"Extracting '{self.data_path}' to '{extraction_folder}' ...")
                with zipfile.ZipFile(self.data_path, 'r') as zip_ref:
                    zip_ref.extractall(path=extraction_folder)
            self.data_path = extraction_folder
        if not self.data_path.is_dir():
            raise ValueError(f"{self.data_path} is not a valid directory.")
        

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
            
        for ax, img_path, cls_name in zip(axes, class_images, class_names):
            img = plt.imread(str(img_path))
            ax.imshow(img)
            ax.set_title(cls_name)
            ax.axis("off")
        
        plt.tight_layout()
        plt.show()

class ModelHandler: 
    def __init__(self, config):
        self.model_config = config["model"]
        # Create an output directory inside "output/" with the model's name.
        self.out_dir = Path("output") / self.model_config["name"]
        os.makedirs(self.out_dir, exist_ok=True)
        # We'll save the best model checkpoint inside this directory.
        self.best_model_path = self.out_dir / "best_model.keras"
        self.image_shape = self.model_config["input_shape"]
        self.model = self.build_model()
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        self.model.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy'])
        # Ensure the log directory is also inside the output folder.
        self.log_dir = self.out_dir / "logs"
        os.makedirs(self.log_dir, exist_ok=True)

    def build_model(self):
        """
        Builds the model using the Functional API based on the configuration.
        """
        inputs = Input(shape=self.image_shape)
        x = inputs
        
        # Build convolutional layers based on configuration.
        for layer_conf in self.model_config["conv_layers"]:
            x = Conv2D(layer_conf["filters"],
                       layer_conf["kernel_size"],
                       padding="same",
                       activation=layer_conf["activation"])(x)
            if layer_conf.get("use_batchnorm", False):
                x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2), padding="same")(x)
        x = GlobalAveragePooling2D()(x)
        
        # Build multiple dense layers as specified in the config
        if "dense_layers" in self.model_config:
            for dense_conf in self.model_config["dense_layers"]:
                x = Dense(dense_conf["units"], activation=dense_conf["activation"], kernel_regularizer=l2(0.001))(x)
                if "dropout" in dense_conf and dense_conf["dropout"] > 0:
                    x = Dropout(dense_conf["dropout"])(x)
        else:
            # Fall back to a single dense layer if none provided.
            x = Dense(self.model_config.get("dense_units", 64), activation="relu")(x)
            x = Dropout(self.model_config.get("dropout_rate", 0.1))(x)
        # Final classification layer
        outputs = Dense(self.model_config["num_classes"], activation="softmax")(x)
        return Model(inputs=inputs, outputs=outputs)
    
    def train(self, data_loader, epochs=10, initial_epoch=0):
        """
        Trains the model using the data provided by the DataLoader.
        """
        tensorboard_callback = TensorBoard(log_dir=str(self.log_dir), update_freq=10)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3)
        
        modelcheckpoint = ModelCheckpoint(
            filepath=str(self.best_model_path),
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='auto'
        )
        
        classifier = self.model.fit(
            data_loader.training_generator,
            epochs=epochs,
            validation_data=data_loader.validation_generator,
            callbacks=[modelcheckpoint, tensorboard_callback, early_stopping, reduce_lr],
            shuffle=True,
            initial_epoch=initial_epoch
        )
        return classifier

    def plot_training_history(self, classifier):
        """Plots and saves the accuracy and loss curves from training."""
        history = classifier.history
        # Determine keys (supports both 'accuracy' and 'acc' naming conventions)
        if 'accuracy' in history:
            acc_key, val_acc_key = 'accuracy', 'val_accuracy'
        elif 'acc' in history:
            acc_key, val_acc_key = 'acc', 'val_acc'
        else:
            print("Accuracy keys not found in history.")
            return

        # Create the figure
        plt.figure()
        plt.plot(history[acc_key], label='Train Accuracy')
        plt.plot(history[val_acc_key], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper left')
        
        acc_save_path = self.out_dir / "accuracy_plot.png"
        plt.savefig(str(acc_save_path))

        plt.figure()

        # Plot Loss
        if 'loss' in history and 'val_loss' in history:
            plt.figure()
            plt.plot(history['loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(loc='upper left')
            loss_save_path = self.out_dir / "loss_plot.png"
            plt.savefig(str(loss_save_path))
        else:
            print("Loss keys not found in history.")


def config(USE_GPU=True):
    """Configures GPU usage."""
    if not USE_GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            print("GPU(s) detected:")
            for device in physical_devices:
                print(" -", device)
        else:
            print("No GPU detected, using CPU instead.")


config_dict = {
    "data": {
        "data_path": "donnees.zip",
        "train_subdir": "entrainement",
        "test_subdir": "test",
        "batch_size_training": 32,
        "batch_size_validation": 8
    },
    "augmentation": {
        "rescale": 1. / 255,
        "rotation_range": 30,
        "width_shift_range": 0.15,
        "height_shift_range": 0.15,
        "zoom_range": 0.1,
        "brightness_range": [0.80, 1.20],
        "horizontal_flip": True,
        "validation_split": 0.10
    },
    "model": {
        "name": "10_deep_wide",
        "input_shape": (256, 256, 3),
        "conv_layers": [
            {"filters": 32, "kernel_size": (3, 3), "activation": "leaky_relu", "use_batchnorm": True},
            {"filters": 64, "kernel_size": (3, 3), "activation": "leaky_relu", "use_batchnorm": False},
            {"filters": 128, "kernel_size": (3, 3), "activation": "leaky_relu", "use_batchnorm": True},
            {"filters": 256, "kernel_size": (3, 3), "activation": "leaky_relu", "use_batchnorm": False},
            {"filters": 512, "kernel_size": (3, 3), "activation": "leaky_relu", "use_batchnorm": True},
            {"filters": 1048, "kernel_size": (3, 3), "activation": "leaky_relu", "use_batchnorm": False},
            {"filters": 2048, "kernel_size": (3, 3), "activation": "leaky_relu", "use_batchnorm": True}
        ],
        "dense_layers": [
            {"units": 256, "activation": "relu", "dropout": 0.2},
            {"units": 128, "activation": "relu", "dropout": 0.2},
            {"units": 64, "activation": "relu", "dropout": 0.1}
        ],
        "num_classes": 6
    }
}


def main():
    parser = argparse.ArgumentParser(description="Train model with optional resume.")
    parser.add_argument("--resume", type=str, help="Path to the checkpoint to resume training from")
    parser.add_argument("--epochs", type=int, default=50, help="Total number of epochs to train")
    args = parser.parse_args()

    config(USE_GPU=True)

    # Initialize DataLoader with full config (it will use data and augmentation settings)
    data_loader = DataLoader(config_dict)

    # Optionally display some samples
    # data_loader.display_random_sample()

    model_handler = ModelHandler(config_dict)
    model_handler.model.summary()

    initial_epoch = 0
    if args.resume:
        print("Resuming training from checkpoint:", args.resume)
        model_handler.model = load_model(args.resume)
        checkpoint_name = Path(args.resume).stem
        m = re.search(r'epoch(\d+)', checkpoint_name)
        if m:
            initial_epoch = int(m.group(1))
            print("Resuming from epoch:", initial_epoch)
        else:
            print("Could not determine epoch from checkpoint filename. Resuming from epoch 0.")

    classifier = model_handler.train(data_loader, epochs=args.epochs, initial_epoch=initial_epoch)
    model_handler.plot_training_history(classifier)

    try:
        from evaluation import eval as evaluate_model_function
        print("Starting automatic evaluation...")
        evaluate_model_function(model_path=model_handler.best_model_path)
    except ImportError as e:
        print("Evaluation module not found. Please run evaluation.py separately.")

if __name__ == "__main__":
    main()

