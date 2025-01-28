default_config = {
    # time in milliseconds to allot to training
    "training_time": 1_800_000,
    # the learning rate of the model
    "learning_rate": 0.0003,
    # which logs should be displayed in the console during training
    # 0: no info, 1: info messages, 2: debug messages
    "train_log_policy": 1,
    # in which formats the logs should be saved
    "train_log_formats": ["stdout", "csv", "log", "tensorboard"],

    # determines the wind strength
    "wind_intensity": 0.3,
    # determines how fast the gradient of the wind field changes
    "wind_len_scale": 80,
    # determines if the wind field should be rendered
    "render_wind": True
}
