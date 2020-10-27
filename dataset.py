import tensorflow as tf
from config.load_config import config
from TPU_checker import AUTO, REPLICAS
from load_tfrecords import read_labeled_tfrecord, read_unlabeled_tfrecord, prepare_image
from collections import namedtuple

LoadedDataset = namedtuple("LoadedDataset", ["dataset", "size"])


class Dataset:
    def __init__(self, config, cross_val_scheme=None):
        self.config = config
        self.cross_val_scheme = cross_val_scheme

    def gcs_path(self):
        GCS_PATH_2019 = "gs://kds-776e6d1df0bcc93716aa3114b1f4398cf69da299a60dd8847fc82253"
        GCS_PATH_2020 = "gs://kds-f6007f115d26f340b9fd91be9c149135d2e9fa9f0e14bfd42390c6f2"

        return GCS_PATH_2019, GCS_PATH_2020

    def create_train_val_datasets(self, split_type):
        GCS_PATH_2019, GCS_PATH_2020 = self.gcs_path()
        # train, val list contains the 5 folds of the kfold data
        train_list = []
        val_list = []
        test_list = []

        # loop 5 times
        for (idxTrain, idxVal) in self.cross_val_scheme.split(np.arange(15)):
            # CREATE TRAIN AND VALIDATION SUBSETS
            if split_type == "train":
                files_train = tf.io.gfile.glob(
                    [GCS_PATH_2020 + "/train{:02d}*.tfrec".format(x) for x in idxTrain]
                )
                if self.config["INC2019"]:
                    files_train += tf.io.gfile.glob(
                        [GCS_PATH_2019 + "/train{:02d}*.tfrec".format(x) for x in idxTrain * 2 + 1]
                    )
                    print("#### Using 2019 external data")
                if self.config["INC2018"]:
                    pass
                np.random.shuffle(files_train)
                train_list.append(files_train)

            if split_type == "val":
                files_valid = tf.io.gfile.glob(
                    [GCS_PATH_2020 + "/train{:02d}*.tfrec".format(x) for x in idxVal]
                )
                val_list.append(files_valid)

            if split_type == "test":
                files_test = np.sort(np.array(tf.io.gfile.glob(GCS_PATH_2020 + "/test*.tfrec")))
                # files_test= tf.io.gfile.glob(GCS_PATH_2020 + "/test*.tfrec")
                test_list.append(files_test)

        if split_type == "train":
            return train_list
        if split_type == "val":
            return val_list
        if split_type == "test":
            return test_list

    def cv_fold_files(self, split_type):
        files = self.create_train_val_datasets(split_type)
        return files

    def get_dataset(
        self,
        files,
        augment=True,
        shuffle=False,
        repeat=False,
        labeled=True,
        return_image_names=True,
    ):

        ds_list = []
        for file_list in files:
            ds = tf.data.TFRecordDataset(file_list, num_parallel_reads=AUTO)
            ds = ds.cache()
            ds_size = count_data_items(file_list)

            if repeat:
                ds = ds.repeat()

            if shuffle:
                ds = ds.shuffle(1024 * 8)
                opt = tf.data.Options()
                opt.experimental_deterministic = False
                ds = ds.with_options(opt)

            if labeled:
                ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
            else:
                ds = ds.map(
                    lambda example: read_unlabeled_tfrecord(example, return_image_names),
                    num_parallel_calls=AUTO,
                )

            ds = ds.map(
                lambda img, imgname_or_label: (
                    prepare_image(img, augment=augment, config=self.config),
                    imgname_or_label,
                ),
                num_parallel_calls=AUTO,
            )

            ds = ds.batch(self.config["BATCH_SIZE"] * REPLICAS)
            ds = ds.prefetch(AUTO)
            ds_list.append(LoadedDataset(dataset=ds, size=ds_size))

        return ds_list

    def get_dataset_loader(self, split_type="train", **kwargs):
        files = self.cv_fold_files(split_type)
        default_args = {
            "train": {"augment": True, "shuffle": True, "repeat": True},
            "val": {"augment": True, "shuffle": False, "repeat": False},
            "test": {
                "augment": False,
                "shuffle": False,
                "repeat": True,
                "labeled": False,
                "return_image_names": False,
            },
        }
        get_dataset_args = {**default_args[split_type], **kwargs}
        return self.get_dataset(files, **get_dataset_args)

