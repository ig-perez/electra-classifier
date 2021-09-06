"""
A Transformed-based binary text classifier
==========================================

This script implements an ELECTRA model to determine if a tweet is about a
disaster or not.

This code requires a GPU to run. Also, next packages should be installed:
- tensorflow-gpu==2.4.1
- tensorflow-text==2.4.1
- tf-models-official==2.4.0
- tensorflow-determinism
- numpy
- pandas
- sklearn
- matplotlib

Read more: https://ivanperez.pe/blog/nlp06-learning-from-kaggle-competitions

:Author: Iván G. Pérez
"""

# ........................Reproducibility Settings..............................
SEED = 19

import os
import random
import tensorflow as tf
from numpy.random import seed

PROCESSOR = "GPU" if len(tf.config.list_physical_devices('GPU')) else "CPU"
NUM_CORES = len(tf.config.list_physical_devices('GPU')) if PROCESSOR == "GPU" else len(tf.config.list_physical_devices('CPU'))
CPU_CORES = len(tf.config.list_physical_devices('CPU'))
GPU_CORES = len(tf.config.list_physical_devices('GPU'))

random.seed(SEED)
seed(SEED)
tf.random.set_seed(SEED)

os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

if PROCESSOR == "CPU":
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

# ..............................................................................

import numpy as np
import pandas as pd
import tensorflow_hub as hub
import tensorflow_text as text
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score
from official.nlp import optimization
from sklearn.model_selection import train_test_split

tfhub_handle_preprocess="https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
tfhub_handle_encoder="https://tfhub.dev/google/electra_small/2"
pd.options.display.max_colwidth = 180

# ..............................................................................

def create_datasets(
    trn_path: str,
    prd_path: str,
    buffer_size: int,
    batch_size: int
    ) -> tuple:

    """
    A method that loads training and submission datasets, splits the training
    dataset for cross validation, and vectorizes the splitted datasets
    """

    def _load_dataset(path: str, shuffle: bool) -> pd.DataFrame:
        """
        Loads a dataset from a CSV file and returns a clean and shuffled dataframe
        """

        dataframe = pd.read_csv(path)
        dataframe.replace([np.nan], "", inplace=True)

        if shuffle:
            return dataframe.sample(frac=1)
        else:
            return dataframe

    # ..............................................................................
    
    def _split_training_dataset(
        training_dataset: pd.DataFrame,
        labels_col: str,
        test_size: float,
        validation_size: float
        ) -> tuple:

        """
        Split a training dataset into training, test and validation datasets
        """

        training_labels = training_dataset[labels_col]

        splitted_ds = train_test_split(
            training_dataset,
            training_labels,
            test_size = test_size,
            stratify = training_labels,
            random_state=SEED
        )

        trn_data = splitted_ds[0]
        tst_data = splitted_ds[1]
        trn_lbls = splitted_ds[2]
        tst_lbls = splitted_ds[3]

        if validation_size <= 0:  # Use one set for validation and test
            val_data = tst_data
            val_lbls = tst_lbls

            assert len(trn_data) + len(tst_data) == len(training_dataset)
        else:    
            labels = trn_data[labels_col]
            splitted_ds = train_test_split(
                trn_data,
                labels,
                test_size = validation_size,
                stratify = labels,
                random_state=SEED
            )

            trn_data = splitted_ds[0]
            val_data = splitted_ds[1]
            trn_lbls = splitted_ds[2]
            val_lbls = splitted_ds[3]

            assert len(trn_data) + len(tst_data) + len(val_data) == len(training_dataset)

        return ((trn_data, trn_lbls), (tst_data, tst_lbls), (val_data, val_lbls))
    
    # ..............................................................................
    
    def _vectorize(
        trn_splt,
        trn_lbls,
        tst_splt,
        tst_lbls,
        val_splt,
        val_lbls,
        prd_raw,
        buffer_size,
        batch_size) -> tuple:
        
        # This raw value makes the F1 calculations less expensive
        val_raw = val_splt["keyword"] + "," + val_splt["location"] + "," + val_splt["text"]
        
        vect_trn = tf.data.Dataset.from_tensor_slices((trn_splt["keyword"] + "," + trn_splt["location"] + "," + trn_splt["text"], trn_lbls))
        vect_tst = tf.data.Dataset.from_tensor_slices((tst_splt["keyword"] + "," + tst_splt["location"] + "," + tst_splt["text"], tst_lbls))
        vect_val = tf.data.Dataset.from_tensor_slices((val_raw, val_lbls))
        
        vect_prd = prd_raw["keyword"] + "," + prd_raw["location"] + "," + prd_raw["text"]  # No need to vectorize this
        vect_prd = pd.DataFrame(vect_prd, columns=["text"])
        vect_prd["id"] = prd_raw["id"]
        
        trn_ds = vect_trn.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        tst_ds = vect_tst.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = vect_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return ((trn_ds, trn_lbls), (tst_ds, tst_lbls), (val_ds, val_raw, val_lbls), (vect_prd, prd_raw))

    # ..............................................................................

    trn_raw = _load_dataset(trn_path, shuffle=False)  # Shuffled inside _vectorize
    prd_raw = _load_dataset(prd_path, shuffle=False)
    
    ((trn_splt, trn_lbls),
    (tst_splt, tst_lbls),
    (val_splt, val_lbls)) = _split_training_dataset(
        training_dataset=trn_raw,
        labels_col="target",
        test_size=0.05,
        validation_size=0)
    
    return _vectorize(
        trn_splt,
        trn_lbls,
        tst_splt,
        tst_lbls,
        val_splt,
        val_lbls,
        prd_raw,
        buffer_size,
        batch_size)

def build_classifier_model(
    preprocessing_layer: hub.KerasLayer,
    encoder: hub.KerasLayer,
    drpt_rate: float
    ) -> tf.keras.Model:
    
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
    encoder_inputs = preprocessing_layer(text_input)
    outputs = encoder(encoder_inputs)
    
    net = outputs["pooled_output"]  # The resulting embedding of the [CLS] token -> <-
    net = tf.keras.layers.Dropout(drpt_rate)(net)
    net = tf.keras.layers.Dense(64, activation="relu")(net)
    net = tf.keras.layers.Dropout(drpt_rate)(net)
    net = tf.keras.layers.Dense(32, activation="relu")(net)
    net = tf.keras.layers.Dropout(drpt_rate)(net)
    net = tf.keras.layers.Dense(1, activation=None, name="classifier")(net)
    
    return tf.keras.Model(text_input, net)

def plot_results(
    history: tf.keras.callbacks.History,
    f1_scores: list
    ) -> None:

    plt.figure(figsize=(32, 16))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['binary_accuracy'])  # metric during training
    plt.plot(history.history['val_binary_accuracy'])
    plt.plot(epochs_f1)
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.legend(["trn_bi_acc.", 'val_bi_acc.', "val_f1"])
    plt.ylim(None, 1)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"])  # metric during training
    plt.plot(history.history['val_loss'], '')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["trn_loss", 'val_loss'])
    plt.ylim(0, 2)

    return None

class CustomMetrics(tf.keras.callbacks.Callback):
    # Based on Katherine (Yi) Li post on neptune blog

    def __init__(self, validation):   
        super(CustomMetrics, self).__init__()
        self.validation = validation 
                    
    def on_train_begin(self, logs={}):        
        self.model.val_f1s = []

    def on_epoch_end(self, epoch, logs={}):
        val_ground_truths = self.validation[1]
        val_predictions = tf.sigmoid(self.model.predict(self.validation[0])).numpy().round()  # In current model the output has no activation!
        
        val_f1 = round(f1_score(val_ground_truths, val_predictions), 4)
        self.model.val_f1s.append(val_f1)
        
        print(f"........... epoch's val_f1: {val_f1}")

# ..............................................................................

if __name__ == "__main__":
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    # 1. Workout datasets
    trn_path = "./data/train.csv"
    prd_path = "./data/test.csv"  # submission ds

    (
        (trn_ds, trn_lbls),
        (tst_ds, tst_lbls),
        (val_ds, val_raw, val_lbls),
        (prd_ds, prd_raw)
        ) = create_datasets(trn_path, prd_path, BUFFER_SIZE, BATCH_SIZE)

    # 2. Preprocessing layer
    preprocessing_layer = hub.KerasLayer(
        tfhub_handle_preprocess,
        name="preprocessing")

    # 3. Selecting the model
    bert_encoder = hub.KerasLayer(
        tfhub_handle_encoder,
        trainable=True,
        name="BERT_encoder")

    # 4. Defining the model
    classifier_model = build_classifier_model(
        preprocessing_layer,
        bert_encoder,
        0.2)

    # 5. Preparing for training
    model_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # No activation in the output
    model_metrics = tf.metrics.BinaryAccuracy()
    init_lr = 3e-5
    epochs = 5
    steps_per_epoch = tf.data.experimental.cardinality(trn_ds).numpy()
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)

    model_optimizer = optimization.create_optimizer(
        init_lr=init_lr,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        optimizer_type="adamw"
    )

    # F1: val_raw, why? because othwerise we should batch it
    training_callbacks = [CustomMetrics(validation=(val_raw, val_lbls))]

    classifier_model.compile(
        optimizer=model_optimizer,
        loss=model_loss,
        metrics=model_metrics)

    # 6. Finetuning
    history = classifier_model.fit(
        trn_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=training_callbacks)  # val_ds is used for train_loss & bi_acc

    # 7. Model evaluation
    loss, accuracy = classifier_model.evaluate(tst_ds)

    # F1. round method is enough to determine labels
    test_predictions = tf.sigmoid(classifier_model.predict(tst_ds)).numpy().round()
    test_f1 = f1_score(tst_lbls, test_predictions)
    epochs_f1 = classifier_model.val_f1s

    print(f'\nLoss: {loss}')
    print(f'Accuracy: {accuracy}')
    print(f'F1: {test_f1}\n')

    plot_results(history, epochs_f1)

    # 8. Generate predictions for test set
    prediction = (classifier_model.predict(
        prd_ds["text"],
        batch_size=BATCH_SIZE) > 0.5).astype('int8')

    assert len(prd_ds) == len(prd_raw)

    results = pd.DataFrame(
        {
            "id": prd_ds["id"],
            "target": np.squeeze(prediction)
        }
    ).sort_values(by=["id"])

    results.to_csv("./output/predictions.csv", index=False)

    # 9. Check predictions
    temp_df = [prd_raw["id"], prd_raw["text"], results["id"], results["target"]]
    headers = ["original_id", "original_text", "test_id", "prediction"]
    check = pd.concat(temp_df, axis=1, keys=headers)

    disasters = check[check["prediction"] == 1]
    not_disasters = check[check["prediction"] == 0]

    assert len(disasters) + len(not_disasters) == len(results)

    print(f"Diaster: {len(disasters)}. Non-disaster: {len(not_disasters)}")

    disasters.head(17)
    not_disasters.head(17)