import json
import os.path
import torch
import numpy as np
import torch_config
from torch_unet_model import Unet
from torch_unet_universal import UNET
from torch_image_to_tesor import get_data
from torch_metrics import DiceMetric, DiceBCELoss
from torch.optim import Adam, SGD, RMSprop
from torch.nn import BCEWithLogitsLoss, DataParallel
import time
from tqdm import tqdm

import matplotlib.pyplot as plt
from torchvision import transforms
from skimage import io, transform
from skimage.util import img_as_ubyte
from torchmetrics.classification import BinaryPrecision, BinaryRecall
import math
import pickle as pkl
import optuna
from optuna.samplers import NSGAIISampler
from baumeva.ga import GaData, BinaryPopulation, HyperbolaFitness, TournamentSelection, OnePointCrossover, BinStringMutation, NewGeneration, UniformCrossover



def grid_search_adam():
    total_start = time.time()
    best_model = {'loss': 10000, 'metric': 0, 'precision': 0, 'recall': 0, 'optimizer': 'Adam', 'cnn_depth': None, 'first_channel': None, 'batch_norm': False,
                  'drop_out': False, 'amsgrad': False, 'lr': None, 'moment': None, 'w_decay': None}
    grid_search_list = []
    iterations = torch_config.SEED_COUNTER*len(torch_config.CNN_DEPTH)*len(torch_config.FIRST_CHANNELS)*len(torch_config.BATCH_NORM)*len(torch_config.DROP_OUT)*len(torch_config.AMSGRAD)*len(torch_config.LR)*len(torch_config.MOMENT)*len(torch_config.WEIGHT_DECAY)
    iteration = 0
    for seed in range(torch_config.SEED_COUNTER):
        for cnn_depth in torch_config.CNN_DEPTH:
            for first_channel in torch_config.FIRST_CHANNELS:
                for batch_norm in torch_config.BATCH_NORM:
                    for drop_out in torch_config.DROP_OUT:
                        for amsgrad in torch_config.AMSGRAD:
                            for lr in torch_config.LR:
                                for moment in torch_config.MOMENT:
                                    for w_decay in torch_config.WEIGHT_DECAY:
                                        print(f'Grid search: {iteration+1}/{iterations}')
                                        print(f'Training CNN with params:\ncnn_depth: {cnn_depth}\nfirst_channel:'
                                              f' {first_channel}\nbatch_norm: {batch_norm}\ndrop_out: {drop_out}\namsgrad:'
                                              f' {amsgrad}\nlr: {lr}\nmoment: {moment}\nw_decay: {w_decay}')
                                        unet = UNET(3, first_channel, 1, batch_norm=batch_norm, drop_out=drop_out,
                                                    downhill=cnn_depth, kernel_size=torch_config.KERNEL_SIZE[0], padding=0)
                                        print(sum(p.numel() for p in unet.parameters()))
                                        if torch.cuda.device_count() > 1:
                                            unet = DataParallel(unet)
                                        opt = Adam(unet.parameters(), lr=lr, amsgrad=amsgrad, betas=moment, weight_decay=w_decay)
                                        checkpoint = {'model': unet, 'state_dict': None, 'optimizer': None,
                                                      'loss': 10000, 'metric': 0, 'accuracy': 0, 'precision': 0, 'recall': 0}
                                        history, checkpoint = training(unet, opt, checkpoint)
                                        if best_model['loss'] > checkpoint['loss']:
                                            best_model['loss'] = checkpoint['loss']
                                            best_model['metric'] = checkpoint['metric']
                                            best_model['precision'] = checkpoint['precision']
                                            best_model['recall'] = checkpoint['recall']
                                            best_model['cnn_depth'] = cnn_depth
                                            best_model['first_channel'] = first_channel
                                            best_model['batch_norm'] = batch_norm
                                            best_model['drop_out'] = drop_out
                                            best_model['amsgrad'] = amsgrad
                                            best_model['lr'] = lr
                                            best_model['moment'] = moment
                                            best_model['w_decay'] = w_decay
                                        name_file_art = (f'i-{seed}_art_depth-{cnn_depth}_feats-{first_channel}_'
                                                         f'BN-{batch_norm}_DO-{drop_out}_LR-{lr}_M-{moment}_'
                                                         f'kernel-{torch_config.KERNEL_SIZE[0]}'
                                                         f'_{torch_config.MODEL_NAME}')
                                        if torch_config.IS_SAVE:
                                            torch.save(checkpoint, os.path.join(torch_config.PATH_OUTPUT, name_file_art))
                                            with open(os.path.join(torch_config.PATH_OUTPUT, 'train_valid_loss_' + name_file_art.split(sep='.pth')[0] + '.pickle'),
                                                      'wb') as file:
                                                pkl.dump(history, file)
                                        # plot_loss(history, name_file_art)
                                        print(f'{seed}: {best_model}')
                                        grid_search_dict = dict.fromkeys(['metric', 'precision', 'recall', 'cnn_depth', 'first_channel', 'batch_norm', 'drop_out', 'amsgrad', 'lr', 'moment', 'w_decay'])
                                        grid_search_dict['metric'] = checkpoint['metric']
                                        grid_search_dict['precision'] = checkpoint['precision']
                                        grid_search_dict['recall'] = checkpoint['recall']
                                        grid_search_dict['cnn_depth'] = cnn_depth
                                        grid_search_dict['first_channel'] = first_channel
                                        grid_search_dict['batch_norm'] = batch_norm
                                        grid_search_dict['drop_out'] = drop_out
                                        grid_search_dict['amsgrad'] = amsgrad
                                        grid_search_dict['lr'] = lr
                                        grid_search_dict['moment'] = moment
                                        grid_search_dict['w_decay'] = w_decay
                                        grid_search_list.append(grid_search_dict)
                                        iteration += 1
                                        with open(os.path.join(torch_config.PATH_OUTPUT, 'best_model_adam.json'), 'w',
                                                  encoding='utf-8') as file:
                                            json.dump(best_model, file, indent=4)

    total_stop = time.time()
    print(f'Training stopped, time: {np.round((total_stop - total_start) / 60, 2)} min')

    with open(os.path.join(torch_config.PATH_OUTPUT, 'best_model_adam.json'), 'w', encoding='utf-8') as file:
        json.dump(best_model, file, indent=4)
    #
    # if torch_config.IS_GRIDSEARCH:
    #     with open(os.path.join(torch_config.PATH_OUTPUT, 'grid_search_adam_lr_m.pickle'), 'wb') as file:
    #         pkl.dump(grid_search_list, file)

    return best_model, grid_search_list, history


def obj_nsga(trial):
    # batch_norm = trial.suggest_int('batch_norm', 0, 1)
    drop_out = trial.suggest_int('drop_out', 0, 1)
    # if batch_norm == 0:
        # batch_norm = False
    # else:
        # batch_norm = True
    if drop_out == 0:
        drop_out = False
    else:
        drop_out = True
    first_channel_feats_num = trial.suggest_int('first_channel_feats_num', 2, 64)
    cnn_depth = trial.suggest_int('cnn_depth', 2, 5)
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7, 9, 11])

    unet = UNET(3, first_channel_feats_num, 1, batch_norm=True, drop_out=drop_out,
                downhill=cnn_depth, padding=0, kernel_size=kernel_size)
    total_model_params = sum(p.numel() for p in unet.parameters())

    if torch.cuda.device_count() > 1:
        unet = DataParallel(unet)

    lr = trial.suggest_float('lr', 0.0001, 1)
    moment_1 = trial.suggest_float('moment_1', 0.9, 0.99)
    moment_2 = trial.suggest_float('moment_2', 0.99, 0.9999)
    # moment = (0.9, moment)

    opt = Adam(unet.parameters(), lr=lr, amsgrad=False, betas=(moment_1, moment_2), weight_decay=0)
    checkpoint = {'model': unet, 'state_dict': None, 'optimizer': None,
                  'loss': 10000, 'metric': 0, 'accuracy': 0, 'precision': 0, 'recall': 0}
    history, checkpoint = training(unet, opt, checkpoint)

    return total_model_params, checkpoint['metric']


def training(unet, opt, checkpoint):
    unet = unet.to(torch_config.DEVICE)
    loss_func = DiceBCELoss()  # BCEWithLogitsLoss()

    train_load, test_load, train_num, test_num = get_data()
    train_step = math.ceil(train_num / torch_config.BATCH_SIZE)
    print(f'Train steps: {train_step}')
    test_step = math.ceil(test_num / torch_config.BATCH_SIZE)
    print(f'Test steps: {test_step}')

    history = {'train_metric': [], 'train_loss': [], 'valid_metric': [], 'valid_loss': [], 'valid_accuracy': [],
               'valid_precision': [], 'valid_recall': []}

    # print('Training the network...')
    start_time = time.time()
    epoch_no_improve = 0
    print(f'Training starts on device: {torch_config.DEVICE}')

    # plt.style.use('ggplot')
    # plt.figure()
    # plt.title('Training Loss (BCE + Dice)')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.show()

    for epoch in tqdm(range(torch_config.NUM_EPOCHS)):
        unet.train()

        total_train_loss = 0
        total_test_loss = 0
        total_train_metric = 0
        total_test_metric = 0

        total_precision = 0
        total_recall = 0

        for (i, (x, y)) in enumerate(train_load):
            (x, y) = (x.to(torch_config.DEVICE), y.to(torch_config.DEVICE))

            opt.zero_grad()
            pred = unet(x)

            loss = loss_func.forward(pred, y)
            total_train_loss += loss.item()

            dm = DiceMetric()
            metric = dm.forward(pred, y)
            total_train_metric += metric.item()

            loss.backward()
            opt.step()

            print(
                f'Training batch {i + 1}. Loss: {np.round(loss.item(), 4)}. Metric of Dice: {np.round(metric.item(), 4)}')

        with torch.no_grad():
            unet.eval()

            for (x, y) in test_load:

                (x, y) = (x.to(torch_config.DEVICE), y.to(torch_config.DEVICE))

                pred = unet(x)

                total_test_loss += loss_func.forward(pred, y).item()

                dm = DiceMetric()
                metric = dm.forward(pred, y)
                total_test_metric += metric.item()

                # bp = BinaryPrecision()
                # bp.to(device=torch_config.DEVICE)
                # metric = bp(pred, y)
                # total_precision += metric.item()
                #
                # br = BinaryRecall()
                # br.to(device=torch_config.DEVICE)
                # metric = br(pred, y).to(torch_config.DEVICE)
                # total_recall += metric.item()

        avg_train_loss = total_train_loss / train_step
        avg_test_loss = total_test_loss / test_step
        avg_train_metric = total_train_metric / train_step
        avg_test_metric = total_test_metric / test_step

        # avg_test_precision = total_precision / test_step
        # avg_test_recall = total_recall / test_step

        history['train_loss'].append(avg_train_loss)
        history['valid_loss'].append(avg_test_loss)
        history['train_metric'].append(avg_train_metric)
        history['valid_metric'].append(avg_test_metric)

        # plt.plot(history['train_loss'], label='train_loss')
        # plt.plot(history['valid_loss'], label='valid_loss')
        # plt.legend(loc='upper right')

        # history['valid_precision'].append(avg_test_precision)
        # history['valid_recall'].append(avg_test_recall)

        print(f'EPOCH: {epoch + 1}/{torch_config.NUM_EPOCHS}')
        print(f'train loss: {avg_train_loss}\ntrain metric: {avg_train_metric}\ntest loss: {avg_test_loss}\n'
              f'test metric: {avg_test_metric}') # \ntest precision: {avg_test_precision}\ntest recall: {avg_test_recall}')

        if avg_test_loss < checkpoint['loss']:  # test
            epoch_no_improve = 0
            checkpoint['state_dict'] = unet.state_dict()
            checkpoint['optimizer'] = opt.state_dict()
            checkpoint['loss'] = avg_test_loss  # test
            checkpoint['metric'] = avg_test_metric
            # checkpoint['precision'] = avg_test_precision
            # checkpoint['recall'] = avg_test_recall
        else:
            epoch_no_improve += 1
        if epoch_no_improve >= torch_config.EPOCHS_NO_IMPROVE:  # early stop
            print(f'Early stop: {epoch_no_improve} epochs without improve')
            break
        if checkpoint['metric'] < 0.001:
            break

    stop_time = time.time()
    print(f'Training stopped, time: {np.round((stop_time - start_time) / 60, 2)} min')
    print('-' * 150)
    return history, checkpoint


def unet_run(params: list) -> float:
    # params
    # 0 - learning rate
    # 1 - betas 1
    # 2 - betas 2
    unet = UNET(3, torch_config.FIRST_CHANNELS[0], 1, batch_norm=torch_config.BATCH_NORM[0],
                drop_out=torch_config.DROP_OUT[0], downhill=torch_config.CNN_DEPTH[0],
                kernel_size=torch_config.KERNEL_SIZE[0], padding=0)
    if torch.cuda.device_count() > 1:
        unet = DataParallel(unet)
    opt = Adam(unet.parameters(), lr=params[0], amsgrad=torch_config.AMSGRAD[0], betas=(params[1], params[2]),
               weight_decay=torch_config.WEIGHT_DECAY[0])
    checkpoint = {'model': unet, 'state_dict': None, 'optimizer': None,
                  'loss': 10000, 'metric': 0, 'accuracy': 0, 'precision': 0, 'recall': 0}
    history, checkpoint = training(unet, opt, checkpoint)
    return checkpoint['metric']


def ga_run():
    ga_data = GaData(num_generations=10)
    ocp = BinaryPopulation()
    ocp.set_params(num_individ=15, gens=((0.0001, 0.1, 0.0001), (0.89, 0.99, 0.01), (0.99, 0.9999, 0.0001)),
                   input_population=None)
    ocp.fill()
    ga_data.population = ocp
    fitness_func = HyperbolaFitness(obj_function=unet_run, obj_value=1)
    fitness_func.execute(ga_data)
    ga_data.update()

    selection = TournamentSelection(tournament_size=2)
    crossover = UniformCrossover()
    mutation = BinStringMutation('strong')
    new_generation = NewGeneration('best')

    for i in range(ga_data.num_generations):

        print(f'Generation {i+1}:')
        selection.execute(ga_data)
        crossover.execute(ga_data)
        mutation.execute(ga_data)
        new_generation.execute(ga_data)
        fitness_func.execute(ga_data)
        ga_data.update()
        print(f"Result: {ga_data.best_solution} ")
        with open(os.path.join(torch_config.PATH_OUTPUT, 'best_ga_gens_adam.json'), 'w', encoding='utf-8') as file:
            json.dump(ga_data.best_solution, file, indent=4)

    print(f"Result: {ga_data.best_solution} ")


def plot_loss(history: dict, name_file: str = None):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['valid_loss'], label='valid_loss')
    plt.title('Training Loss (BCE + Dice)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()
    if name_file is not None:
        plt.savefig(os.path.join(torch_config.PATH_OUTPUT, name_file + '.png'), dpi=500)
        plt.close()


if __name__ == '__main__':
    torch.cuda.empty_cache()
    import random
    torch.manual_seed(19)
    torch.cuda.manual_seed_all(19)
    random.seed(19)
    np.random.seed(19)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(19)

    hyper_best_dict, hyper_total_dict, history_train = grid_search_adam()
    plot_loss(history=history_train, name_file='bright_exp_1')

    # ga_run()

    # hyper_best_dict, hyper_total_dict = grid_search_rmsprop()
    # hyper_dict = grid_search_sgd()
    # history_1, best_dict = training()
    # plot_loss(history=history_train)
    # with open('/home/ii/SegRat/Bright_data/output/exp_0/train_valid_loss_i-0_art_depth-4_feats-32_BN-True_DO-True_LR-0.001_M-(0.9, 0.999)_unet_Adam_bright_data_0.pickle', 'rb') as f:
    #     history_dict = pkl.load(f)
    #
    # print(history_dict.keys())
    # plot_loss(history_dict, 'bright_exp_0')
    """
    study = optuna.create_study(sampler=NSGAIISampler(population_size=20, seed=5, mutation_prob=0.1,
                                                      ),
                                directions=['minimize', 'maximize'])
    study.optimize(obj_nsga, n_trials=200, timeout=1200000)
    print('Best trials:')
    trials = sorted(study.best_trials, key=lambda t: sum(t.values))
    for trial in trials:
        print(f"Trial: {trial.number} Values: {trial.values} Params: {trial.params}\n")

    print('Total trials:')
    trials_total = sorted(study.trials, key=lambda t: sum(t.values))
    for trial in trials_total:
        print(f"Trial: {trial.number} Values: {trial.values} Params: {trial.params}\n")
    trials_to_save = {'best_trials': trials, 'total_trials': trials_total}
    with open(os.path.join(torch_config.PATH_OUTPUT, 'valid_trials_NSGA2.pickle'), 'wb') as file:
        pkl.dump(trials_to_save, file)
    """
