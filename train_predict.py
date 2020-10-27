import efficientnet.tfkeras as efn
from sklearn.model_selection import KFold
import tensorflow.keras.backend as K
import tensorflow as tf
from config.load_config import config
from dataset import Dataset
from load_tfrecords import count_data_items
from dataset import get_dataset
from TPU_checker import REPLICAS

# LEARNING RATE SCHEDULER
def get_lr_callback(config):
    lr_start = config["LR_START"]
    lr_max = config["LR_MAX"] * REPLICAS
    lr_min = config["LR_MIN"]
    lr_ramp_ep = config["LR_RAMPUP_EPOCHS"]
    lr_sus_ep = config["LR_SUSTAIN_EPOCHS"]
    lr_decay = config["LR_EXP_DECAY"]

    def lr_function(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start

        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max

        else:
            lr = (lr_max - lr_min) * lr_decay ** (epoch - lr_ramp_ep - lr_sus_ep) + lr_min

        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_function, verbose=False)
    return lr_callback


class Model:
    def __init__(self, config):
        self.config = config

    def build_model(self):
        model_input = tf.keras.layers.Input(
            shape=(self.config["IMG_SIZES"], self.config["IMG_SIZES"], 3)
        )
        base_model = getattr(efn, self.config["EFF_NET"])(
            input_shape=(self.config["IMG_SIZES"], self.config["IMG_SIZES"], 3),
            weights="imagenet",
            include_top=False,
        )
        x = base_model(model_input)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs=model_input, outputs=x)
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)
        model.compile(optimizer=opt, loss=loss, metrics=["AUC"])
        return model

    def run_fold(self, train_loader, val_loader, fold=None):
        if self.config["DEVICE"] == "TPU":
            if tpu:
                tf.tpu.experimental.initialize_tpu_system(tpu)

        # BUILD MODEL
        K.clear_session()
        with strategy.scope():
            model = self.build_model()

        # SAVE BEST MODEL EACH FOLD
        save_best_model = tf.keras.callbacks.ModelCheckpoint(
            "fold-{}.h5".format(fold),
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            save_freq="epoch",
        )

        # TRAIN
        print("Training...")
        history = model.fit(
            train_loader[fold].dataset,
            epochs=self.config["EPOCHS"],
            callbacks=[save_best_model, get_lr_callback(self.config)],
            steps_per_epoch=train_loader[fold].size / self.config["BATCH_SIZE"] // REPLICAS,
            validation_data=val_loader[fold].dataset,
            verbose=1,
        )

    def run(self, train_loader, val_loader, all_folds=False, fold_num=None):
        # try except here to improve the code.
        # reason for fold and cross val to be none is that both of them may or may not be needed
        if all_folds:
            for fold in range(5):
                self.run_fold(train_loader, val_loader, fold)
        else:
            self.run_fold(train_loader, val_loader, fold_num)

    # this should only be called after running the training
    def predict(self, test_loader, weight_path, tta=True):

        if self.config["DEVICE"] == "TPU":
            if tpu:
                tf.tpu.experimental.initialize_tpu_system(tpu)

        # BUILD MODEL
        K.clear_session()
        with strategy.scope():
            model = self.build_model()

        print("Loading best model...")
        model.load_weights(weight_path)

        # TEST, somehow need to have batch size 64 for all, including loading test set.
        if tta:
            # PREDICT TEST USING TTA
            print("Predicting Test with TTA...")

            config_copy = self.config.copy()
            config_copy["BATCH_SIZE"] = config_copy["BATCH_SIZE"] * 4
            test_size = test_loader.size
            preds = np.zeros((test_size, 1))
            STEPS = (
                (config_copy["TTA"] * test_size) / (config_copy["BATCH_SIZE"] / 4) / 4 / REPLICAS
            )
            pred = model.predict(test_loader.dataset, steps=STEPS, verbose=1)[
                : config_copy["TTA"] * test_size,
            ]
            preds[:, 0] += (
                np.mean(pred.reshape((test_size, config_copy["TTA"]), order="F"), axis=1) * 0.2
            )

        else:
            config_copy = self.config.copy()
            config_copy["BATCH_SIZE"] = config_copy["BATCH_SIZE"] * 4
            test_size = test_loader.size
            STEPS = (
                (config_copy["TTA"] * test_size) / (config_copy["BATCH_SIZE"] / 4) / 4 / REPLICAS
            )
            preds = model.predict(test_loader.dataset, steps=STEPS, verbose=1)[
                :test_size,
            ]

        return preds


def run_training():
    fitter = Model(config=config)
    dataset = Dataset(config, cross_val_scheme=skf)
    train_loader1 = dataset.get_dataset_loader("train")
    val_loader1 = dataset.get_dataset_loader("val")
    # train
    fitter.run(train_loader1, val_loader1, all_folds=False, fold_num=3)


def run_prediction():
    fitter = Model(config=config)
    dataset = Dataset(config, cross_val_scheme=skf)
    # note here config test is different from train val because if we use tta need make config diff

    config_test = config.copy()
    config_test["BATCH_SIZE"] = config_test["BATCH_SIZE"] * 4
    dataset_test = Dataset(config_test, cross_val_scheme=skf)
    test_loader = dataset_test.get_dataset_loader("test", augment=False)[0]
    # train
    predictions_without_tta = fitter.predict(
        test_loader, weight_path="../input/testsiim/finale.h5", tta=False
    )
    predictions_with_tta = fitter.predict(
        test_loader, weight_path="../input/testsiim/finale.h5", tta=True
    )
    return predictions_without_tta, predictions_with_tta


def submission():

    fitter = Model(config=config)
    dataset_test = Dataset(config, cross_val_scheme=skf)
    # note here config test is different from train val because if we use tta need make config diff
    test_loader_with_img_name = dataset_test.get_dataset_loader(
        "test", augment=False, repeat=False, labeled=False, return_image_names=True
    )[0]
    image_names = np.array(
        [
            img_name.numpy().decode("utf-8")
            for img, img_name in iter(test_loader_with_img_name.dataset.unbatch())
        ]
    )

    submission = pd.DataFrame(dict(image_name=image_names, target=predictions_without_tta[:, 0]))
    submission = submission.sort_values("image_name")
    submission.to_csv("submission_without_tta.csv", index=False)
    submission.head()


if __name__ == "__main__":
    run_training()
    predictions_without_tta, predictions_with_tta = run_prediction()
    submission()
