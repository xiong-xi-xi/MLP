import matplotlib.pyplot as plt
import os
from datetime import datetime

save_dir = 'figure'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

def draw_loss(train_loss_list, hyperparam_str):
    plt.plot(train_loss_list)
    plt.xlabel('epoch')
    plt.ylabel('train loss')
    title = '\n'
    title += 'Hyperparameters: \n'
    title += hyperparam_str + '\n'
    plt.title(title, fontsize=12)
    dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(os.path.join(save_dir, dt) + 'TrainLoss', bbox_inches='tight')
    plt.close()

def draw_acc(train_acc_list, best_train_test_acc_str, hyperparam_str):
    plt.plot(train_acc_list)
    plt.xlabel('epoch')
    plt.ylabel('train acc')
    title = '\n'
    title += 'Hyperparameters: \n'
    title += hyperparam_str + '\n'
    title += '(BestTrainAcc, BestTestAcc): '
    title += best_train_test_acc_str + '\n'
    plt.title(title, fontsize=12)
    dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    plt.savefig(os.path.join(save_dir, dt) + 'TrainTestAcc', bbox_inches='tight')
    plt.close()
