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
