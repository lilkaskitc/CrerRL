import tensorflow as tf
from tensorflow import keras
from keras.optimizers import SGD, Adam
from keras import datasets
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time
import argparse
import yaml
import logging
import pickle

from models.crenetRESMultiGPUs import CreNetRES50

def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
    return config

def single_model_evaluate(model, x_test, y_test):
    
    pred = model.predict(x_test)
    
    m = tf.keras.metrics.CategoricalAccuracy()
    
    C = int(pred.shape[-1]/2)
    
    m.update_state(y_test, pred[:,:C])
    acc_L = m.result().numpy()
    m.reset_state()
    
    m.update_state(y_test, pred[:,C:])
    acc_U = m.result().numpy()
    m.reset_state()
      
    return pred, acc_L, acc_U
    
def main():
    strategy = tf.distribute.MirroredStrategy()
    print('Number of GPU devices: {}'.format(strategy.num_replicas_in_sync))
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    # Accept a YAML file as a command-line argument
    parser = argparse.ArgumentParser(description='Process parameters from a YAML file.')
    parser.add_argument('config_file', type=str, help='Path to the YAML configuration file')
    args = parser.parse_args()

    config = load_config(args.config_file)

    # Access hyperparameters from the loaded configuration
    seed = config['Seed']
    print('Applied Seed :', seed)
    delta = 0.5
    batch_size = config['BatchSize']
    epochs = 40
    # epochs = 50
    
    learning_rate = config['LearningRate']
    pre_weights = config['PreWeights']
    # verbose = config['Verbose']
    verbose = True
    # es = config['EarlyStopping']
    print('Delta: ', delta)
    
    # Define the save path
    full_path = 'train_results100Res/'+str(delta)+'/'+str(seed)
    full_path_his = 'train_results100Res/'+str(delta)+'/his/'+str(seed)
    
    # Set random seed
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    # Define Learning Scheduler 
    def lr_scheduler(epoch):
        """Learning Rate Schedule
    
        Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
        Called automatically every epoch as part of callbacks during training.
    
        # Arguments
            epoch (int): The number of epochs
    
        # Returns
            lr (float32): learning rate
        """
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        # elif epoch > 80:
        # elif epoch >= 45:
        elif epoch >= 35:
        # elif epoch > 30:
        # elif epoch > 20:
            lr *= 1e-1
        return lr
    lr_scheduler_mod = lr_scheduler
    ###################################################################
    ##################### Prepare training dataset ####################
    ###################################################################
    
    (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
    
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = to_categorical(y_train, 100)
    y_test = to_categorical(y_test, 100)
    
    # standard normalizing
    x_train = (x_train - np.array([[[0.4914, 0.4822, 0.4465]]])) / np.array([[[0.2023, 0.1994, 0.2010]]])
    x_test = (x_test - np.array([[[0.4914, 0.4822, 0.4465]]])) / np.array([[[0.2023, 0.1994, 0.2010]]])
    
    val_samples = -10000
    
    x_val = x_train[val_samples:]
    y_val = y_train[val_samples:]
    
    
    x_train = x_train[:val_samples]
    y_train = y_train[:val_samples]
    
    datagen = ImageDataGenerator(zca_epsilon=1e-06, width_shift_range=0.1, height_shift_range=0.1, fill_mode='nearest',horizontal_flip=True)
    datagen.fit(x_train)
    
    # augmented_data = datagen.flow(x_train, y_train, batch_size=len(x_train), shuffle=False)
    
    # Get the processed data (x_train) and labels
    # x_train_augmented, y_train_augmented = augmented_data.next()
    x_train_augmented = x_train
    y_train_augmented = y_train
    # print("Shape of x_train_augmented:", x_train_augmented.shape)
    
    BUFFER_SIZE = len(x_train_augmented)

    BATCH_SIZE_PER_REPLICA = batch_size
    
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_augmented, y_train_augmented)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(GLOBAL_BATCH_SIZE)


    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

    ###################################################################
    ####################### Define CreNet Model #######################
    ###################################################################
    with strategy.scope():
        model = CreNetRES50(input_shape=(32, 32, 3), classes=100, weights='imagenet')
        model.compile(optimizer=Adam(learning_rate=learning_rate))
        
    # Define the metrics
    with strategy.scope():
        
        total_loss_tracker = keras.metrics.Mean(name="Loss-T")
        upper_loss_tracker = keras.metrics.Mean(name="Loss-U")
        lower_loss_tracker = keras.metrics.Mean(name="Loss-L")
        val_upper_loss_tracker = keras.metrics.Mean(name="ValLoss-U")
        val_lower_loss_tracker = keras.metrics.Mean(name="ValLoss-L")
        
        upper_acc_tracker = keras.metrics.CategoricalAccuracy(name='Acc-U')
        lower_acc_tracker = keras.metrics.CategoricalAccuracy(name='Acc-L')
        val_upper_acc_tracker = keras.metrics.CategoricalAccuracy(name='ValAcc-U')
        val_lower_acc_tracker = keras.metrics.CategoricalAccuracy(name='ValAcc-L')
        
    ###################################################################
    ####################### Define Train Step #########################
    ###################################################################
    def train_step(data):
    
        inputs, labels = data
        train_batch_num = labels.shape[0]
        with tf.GradientTape() as tape:
            preds = model(inputs, training=True)
    
            # Extract upper and lower probabilities
            preds_lo = preds[:, :labels.shape[-1]]
            preds_up = preds[:, labels.shape[-1]:]
    
            # Compute loss related to lower probabilities

            loss_lo = tf.keras.losses.CategoricalCrossentropy(
                reduction=tf.keras.losses.Reduction.NONE)(labels, (preds_lo))

            # Select top delta * batch_size samples with highest loss for backward
            loss_lo_sort = tf.sort(loss_lo, direction='DESCENDING', axis=-1)

            bound_index = int(np.floor(delta*train_batch_num))-1
            bound_value = loss_lo_sort[bound_index]

            choose_index = tf.greater_equal(loss_lo, bound_value)
            choose_preds_lo = preds_lo[choose_index]
            choose_labels = labels[choose_index]

            loss_lo_mod = tf.reduce_mean(
                    tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
                    (choose_labels, (choose_preds_lo)))

            loss_up = tf.reduce_mean(
                tf.keras.losses.CategoricalCrossentropy(
                    reduction=tf.keras.losses.Reduction.NONE)(labels, (preds_up)))
            
            loss_total = loss_lo_mod + loss_up
    
        grads = tape.gradient(loss_total, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
        total_loss_tracker.update_state(loss_total)
        upper_loss_tracker.update_state(loss_up)
        lower_loss_tracker.update_state(loss_lo_mod)
        upper_acc_tracker.update_state(labels, preds_up)
        lower_acc_tracker.update_state(labels, preds_lo) 
        return loss_total

    ###################################################################
    ######################## Define Test Step #########################
    ###################################################################
    def test_step(data):
        inputs, labels = data
        preds = model(inputs, training=False)

        # Extract upper and lower probabilities
        preds_lo = preds[:, :labels.shape[-1]]
        preds_up = preds[:, labels.shape[-1]:]

        # Compute the relavant loss using upper and lower probabilities
        loss_lo = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)(labels, preds_lo)

        loss_up = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)(labels, preds_up)
          
        val_upper_loss_tracker.update_state(loss_up)
        val_lower_loss_tracker.update_state(loss_lo)
        
        # Update validation accuracy
        val_upper_acc_tracker.update_state(labels, preds_up)
        val_lower_acc_tracker.update_state(labels, preds_lo)



    # `run` replicates the provided computation and runs it
    # with the distributed input.
    @tf.function
    def distributed_train_step(dataset_inputs):
      per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
      return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                             axis=None)
    # def distributed_train_step(dataset_inputs, train_batch_num):
    #   per_replica_losses = strategy.run(train_step, args=(dataset_inputs, train_batch_num))
    #   return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
    #                          axis=None)    
    @tf.function
    def distributed_test_step(dataset_inputs):
      return strategy.run(test_step, args=(dataset_inputs,))
    
    ###################################################################
    ######################## Train & Validate Loop ####################
    ###################################################################
    start = time.time()
    result_history = {'Acc-U': [], 'Acc-L': [], 'Loss-U': [], 'Loss-L': [], 'val_Acc-U': [], 'val_Acc-L': [], 'val_Loss-U': [], 'val_Loss-L': []}
    for epoch in range(epochs):
        
        if lr_scheduler_mod is not None:
           lr = lr_scheduler_mod(epoch)
           model.optimizer.learning_rate.assign(lr)
        
        # TRAIN LOOP
        total_loss = 0.0
        num_batches = 0
        for x in train_dist_dataset:
            distributed_train_step(x)
        result_history['Acc-U'].append(upper_acc_tracker.result().numpy())
        result_history['Acc-L'].append(lower_acc_tracker.result().numpy())
        result_history['Loss-U'].append(upper_loss_tracker.result().numpy())
        result_history['Loss-L'].append(lower_loss_tracker.result().numpy())


        # TEST LOOP
        for x in test_dist_dataset:
            distributed_test_step(x)
        
        # Update to history per epoch
        result_history['val_Acc-U'].append(val_upper_acc_tracker.result().numpy())
        result_history['val_Acc-L'].append(val_lower_acc_tracker.result().numpy())
        result_history['val_Loss-U'].append(val_upper_loss_tracker.result().numpy())
        result_history['val_Loss-L'].append(val_lower_loss_tracker.result().numpy())
    
        template = ("Epoch {}, LossU: {:.4f}, LossL: {:.4f}, AccU: {:.4f}, AccL: {:.4f}, " 
                    "TestLossU: {:.4f}, TestLossL: {:.4f}, TestAccU: {:.4f}, TestAccL: {:.4f}")
        
        # if verbose:
        print(template.format(epoch + 1, upper_loss_tracker.result(), lower_loss_tracker.result(), upper_acc_tracker.result(), lower_acc_tracker.result(), val_upper_loss_tracker.result(), val_lower_loss_tracker.result(), val_upper_acc_tracker.result(), val_lower_acc_tracker.result()))
    
        upper_acc_tracker.reset_states()
        lower_acc_tracker.reset_states()
        upper_loss_tracker.reset_states()
        lower_loss_tracker.reset_states()
        val_upper_acc_tracker.reset_states()
        val_lower_acc_tracker.reset_states()
        val_upper_loss_tracker.reset_states()
        val_lower_loss_tracker.reset_states()
    
        
    end = time.time()
    print(end-start)   
    result = result_history

    # Save trainig history
    with open(full_path_his + '_result', 'wb') as file:
        pickle.dump(result, file)
        
    weights_to_save = model.get_weights() 
    with open(full_path + '_weights', 'wb') as file2:
        pickle.dump(weights_to_save, file2)
    
    # model.save(full_path+'_model.keras')
    
    # creNetTest = CreNetRES50(input_shape=(32, 32, 3), classes=100, weights='imagenet')
    # creNetTest.compile(optimizer=Adam(learning_rate=learning_rate))
    
    # with open(full_path + '_weights', 'rb') as file3:
    #     weights = pickle.load(file3)
        
    # creNetTest.set_weights(weights)
    
    # np.testing.assert_allclose(model.predict(x_test), creNetTest.predict(x_test))

    pred, acc_L, acc_U = single_model_evaluate(model, x_test, y_test)
    
    print('acc_L: ', acc_L)
    print('acc_U: ', acc_U)

if __name__ == "__main__":
    main()
    