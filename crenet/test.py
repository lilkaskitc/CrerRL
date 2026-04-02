import pickle
import numpy as np
import tensorflow as tf
import argparse
import yaml
from keras.optimizers import Adam
from test_core.load_test_data import load_cifar_corruption, load_cinic10_test, load_svhn_test, load_cifar10, load_tinyimage_test
from models.bare_crenet import crenet_res50
from models.crenetRESMultiGPUs import CreNetRES50, CreNetVITBase
import time

def load_model(delta, path, learning_rate=0.001):
    opt=Adam(learning_rate=learning_rate)
    creNet = crenet_res50(input_shape=(32, 32, 3), num_classes=10)
    # creNet = CreNetRES50((32, 32, 3), 10, weights=None)
    # creNet = CreNetVITBase(input_shape=(32, 32, 3), classes=10, weights='imagenet')
    creNet.compile(optimizer=opt)
    
    with open(path + '_weights', 'rb') as file:
        weights = pickle.load(file)
        
    creNet.set_weights(weights)
    return creNet

def single_model_evaluate(model, x_test, y_test, IFAcc):
    
    pred = model.predict(x_test)
    
    if IFAcc:
        m = tf.keras.metrics.CategoricalAccuracy()
        m.update_state(y_test, pred[:,:y_test.shape[-1]])
        acc_L = m.result().numpy()
        m.reset_state()
    
        m.update_state(y_test, pred[:,y_test.shape[-1]:])
        acc_U = m.result().numpy()
        m.reset_state()
        print('acc_U:', acc_U)
        
    else:
        acc_L = None
        acc_U = None
    return pred, acc_L, acc_U

def crenet_evaluation(delta, run_num, dataset):
    seeds = [0, 66, 99, 314, 524, 803, 888, 908, 1103, 1208, 7509, 11840, 40972, 46857, 54833]

        
    ######### Load models #########
    model_list = list()
    for i in range(15):
        
        # model_path = 'train_results32RES10/'+str(delta)+'/'+str(seeds[i])
        # model_path = 'train_resultsRES10/'+str(delta)+'/'+str(seeds[i])
        model_path = '10train_RES50/'+str(delta)+'/'+str(seeds[i])
        crenet = load_model(delta, model_path)
        model_list.append(crenet)

    ######## Unified dictionary for saving the results ########
    cifar = dict()

    if dataset == 'CIFAR10':
        ######### Test on CIFAR10 dataset #########
        cifar = {'pred': [], 'acc_L': [], 'acc_U': [], 'entro': [], 'gh': [], 'label': []}
        
        (_, _), (x_cifar, y_cifar) = load_cifar10()
        ################################
        ######### Single Model #########
        ################################
        for i in range(15):
            model = model_list[i]
            pred, acc_L, acc_U = single_model_evaluate(model, x_cifar, y_cifar, IFAcc=True)
            # Save the result of single model
            cifar['pred'].append(pred)
            cifar['acc_L'].append(acc_L)
            cifar['acc_U'].append(acc_U)    

        cifar['label'] = y_cifar

    elif 'CIFAR10C' in dataset:
        ######### Test on CIFAR10_C dataset #########  
        ######### Determine Corruption Type and Severity Level #########
        parts = dataset.split('_')
        if len(parts) == 3:
            cor_type = int(parts[1])
            sev_level = int(parts[2])
        else:
            print("Invalid dataset string format.")
                            
        x_test, y_test = load_cifar_corruption(cor_type, sev_level)
        
        cifar = {'pred': [], 'acc_L': [], 'acc_U': [], 'entro': [], 'gh': []}
        cifar3 = {'pred': [], 'acc_L': [], 'acc_U': [], 'entro': [], 'gh': []}
        cifar5 = {'pred': [], 'acc_L': [], 'acc_U': [], 'entro': [], 'gh': []}
        for i in range(15):
            model = model_list[i]
            pred, acc_L, acc_U = single_model_evaluate(model, x_test, y_test, IFAcc=True)

            # Save the result of single model
            cifar['pred'].append(pred)
            cifar['acc_L'].append(acc_L)
            cifar['acc_U'].append(acc_U)

    elif dataset == 'SVHN':
        ######### Test on SVHN dataset #########
        cifar = {'pred': [], 'entro': [], 'gh': []}
        cifar3 = {'pred': [], 'entro': [], 'gh': []}
        cifar5 = {'pred': [], 'entro': [], 'gh': []}

        x_svhn, y_svhn = load_svhn_test()
        
        for i in range(15):           
            model = model_list[i]
            pred, _, _, = single_model_evaluate(model, x_svhn, y_svhn, IFAcc=False)

            # Save the result of single model
            cifar['pred'].append(pred)

    elif dataset == 'TinyImage':
        ######### Test on CIFAR10 dataset #########
        cifar = {'pred': [], 'entro': [], 'gh': []}
        cifar3 = {'pred': [], 'entro': [], 'gh': []}
        cifar5 = {'pred': [], 'entro': [], 'gh': []}

        x_tiny, y_tiny = load_tinyimage_test()
        ################################
        ######### Single Model #########
        ################################
        for i in range(15):
            model = model_list[i]
            pred, acc_L, acc_U = single_model_evaluate(model, x_tiny, y_tiny, IFAcc=False)
            # Save the result of single model
            cifar['pred'].append(pred)
    else:
        print("Invalid Dataset Name. Try again...")
    return cifar


def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
    return config


def main():
    
    # Accept a YAML file as a command-line argument
    parser = argparse.ArgumentParser(description='Process parameters from a YAML file.')
    parser.add_argument('config_file', type=str, help='Path to the YAML configuration file')
    args = parser.parse_args()

    config = load_config(args.config_file)

    # Access hyperparameters from the loaded configuration
    # exp_num = config['ExpNum']
    exp_num = 1
    dataset_name = config['Dataset']
    delta = config['Delta']
    # delta = 0.625
    # delta = 0.75
    # delta = 0.875
    
    start_time = time.time()
    result = crenet_evaluation(delta, exp_num, dataset_name)
    end_time = time.time()
    print(end_time - start_time)
    
    full_path = 'test_results32RES10/' + str(delta) + '/' + dataset_name
    # full_path = 'test_resultsRES10/' + str(delta) + '/' + dataset_name

    # Save test history
    with open(full_path + '_result', 'wb') as file:
        pickle.dump(result, file)

if __name__ == "__main__":
    main()
