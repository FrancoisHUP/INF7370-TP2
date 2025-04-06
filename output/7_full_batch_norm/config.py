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
        "name": "7_full_batch)",
        "input_shape": (128, 128, 3),
        "conv_layers": [
            {"filters": 32, "kernel_size": (3, 3), "activation": "leaky_relu", "use_batchnorm": True},
            {"filters": 64, "kernel_size": (3, 3), "activation": "leaky_relu", "use_batchnorm": True},
            {"filters": 128, "kernel_size": (3, 3), "activation": "leaky_relu", "use_batchnorm": True},
            {"filters": 256, "kernel_size": (3, 3), "activation": "leaky_relu", "use_batchnorm": True},
            {"filters": 512, "kernel_size": (3, 3), "activation": "leaky_relu", "use_batchnorm": True}
        ],
        # Instead of a single dense layer, we now allow a list.
        "dense_layers": [
            {"units": 128, "activation": "relu", "dropout": 0.2},
            {"units": 64, "activation": "relu", "dropout": 0.1}
        ],
        "num_classes": 6
    }
}