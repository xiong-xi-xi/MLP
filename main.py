from MultiLayerPerceptron import *
from load_data import *
from utils import *
from args import *
from draw_tools import *

def get_hyper(hyper):
    return (hyper['layer_dim'], hyper['init_method'], hyper['weight_scale'],
           hyper['activate_func'], hyper['loss_func'], hyper['lr'], hyper['regularization'],  hyper['l_lambda'])

def train(model, train_data, test_data):
    train_loss_plot = []
    train_acc_plot = []

    best_train_acc = -99
    best_test_acc = -99

    train_img, train_label = train_data
    test_img, test_label = test_data

    train_size = train_img.shape[0]

    train_loss_list = []
    train_acc_list = []
    batch_num = int(train_size * 1.0 / batch_size)

    for i in range(epoch):
        for j in range(batch_num):
            start = j * batch_size
            end = (j+1) * batch_size
            batch_img = train_img[start : end]
            batch_label = train_label[start : end]
            out, loss = model.forward(batch_img, batch_label)
            train_acc = Accuracy(out, batch_label)

            model.backward(out, batch_label)

            train_loss_list.append(loss)
            train_acc_list.append(train_acc)

        train_loss_mean = np.mean(train_loss_list)
        train_acc_mean = np.mean(train_acc_list)

        train_loss_plot.append(train_loss_mean)
        train_acc_plot.append(train_acc_mean)

        if(train_acc_mean > best_train_acc):
            best_train_acc = train_acc_mean

        if(i%10 == 0):
            print(f'train epoch:{i} ok !')
        #print(f'train--- epoch: {i}, train loss: {train_loss_mean}, train acc: {train_acc_mean}')

        if((i+1) % 5 == 0):
            out, test_loss = model.forward(test_img, test_label)
            test_acc = Accuracy(out, test_label)
            if test_acc > best_test_acc:
                best_test_acc = test_acc

            #print(f'test--- epoch: {i}, test acc: {test_acc}')
    print("train done!")
    best_train_test_acc_str = '(' + '%.2f' % (best_train_acc) + ' ,' + '%.2f' % (best_test_acc) + ')'
    draw_loss(train_loss_plot, hyperparam_str)
    draw_acc(train_acc_plot, best_train_test_acc_str, hyperparam_str)

if __name__ == "__main__":

    for i in range(len(hyper_dict)):
        (layer_dim, init_method, weight_scale, activate_func, loss_func, lr, regularization, l_lambda) = get_hyper(hyper_dict[i])

        MLP_net = \
            Multi_Layer(layer_dim_list = layer_dim, init_method = init_method, weight_scale = weight_scale,
                        activate_func = activate_func, loss_func = loss_func, lr = lr, regularization = regularization, l_lambda = l_lambda)

        hyperparam_str = "init: "+ str(init_method) +", " + \
         'lr: '+ str(lr) + ', ' + \
          'L' + str(regularization)+'-regular'+ ', ' + \
        'lambda: ' + str(l_lambda) + ',\n' + \
        'activate: ' + str(activate_func)+ ', ' + \
        'loss_f: ' + str(loss_func)+ '.'

        print(hyperparam_str)
        print("="*100)

        train_data, test_data = load_dataset()
        train(MLP_net, train_data, test_data)

