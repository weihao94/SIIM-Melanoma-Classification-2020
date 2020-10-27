import os
import yaml


# folder to load config file
CONFIG_PATH = "./config"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


# load config.yaml
config = load_config("config.yaml")
# print(config)
# config as follows
{
    "DEVICE": "GPU",
    "SEED": 42,
    "FOLDS": 5,
    "IMG_SIZES": 768,
    "INC2018": 0,
    "INC2019": 0,
    "BATCH_SIZE": 1,
    "EPOCHS": 12,
    "EFF_NET": "EfficientNetB3",
    "TTA": 11,
    "ROT_": 180.0,
    "SHR_": 1.5,
    "HZOOM_": 6.0,
    "WZOOM_": 6.0,
    "HSHIFT_": 6.0,
    "WSHIFT_": 6.0,
    "CROP_SIZE": {256: 250, 384: 370, 512: 500, 768: 750},
    "LR_START": 3e-06,
    "LR_MAX": 2e-05,
    "LR_MIN": 1e-06,
    "LR_RAMPUP_EPOCHS": 5,
    "LR_SUSTAIN_EPOCHS": 0,
    "LR_EXP_DECAY": 0.8,
    "OPTIMIZER": "tf.keras.optimizers.Adam(learning_rate=0.001)",
    "LOSS": "tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)",
}
