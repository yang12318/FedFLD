from utils import util_dataset, util_parser
import torch
import os
import random
import numpy as np
from models import model_choose_fn
from methods import FedFLD
import matplotlib.pyplot as plt


def run(conf):
    print('Init-------------------------')
    root_path = os.getcwd()
    if root_path.endswith('scripts'):
        root_path = os.path.dirname(root_path)

    conf['savepath'] = os.path.join(root_path, conf['savepath'].strip())
    print('Data and results save path is: ', conf['savepath'])
    ######################################################
    # Provide reproducibility
    torch.manual_seed(conf['seed'])
    random.seed(conf['seed'])
    np.random.seed(conf['seed'])
    in_channel = 3
    out_channel = 10
    ######################################################
    # Split the dataset
    print('Data split-------------------------')
    data_obj = util_dataset.DatasetObject(dataset=conf['dataset'],
                                           n_client=conf['n_client'],
                                           seed=conf['seed'],
                                           rule=conf['rule'],
                                           rule_arg=conf['alpha'],
                                           data_path=conf['savepath'].strip())

    ######################################################
    # Model selection
    if conf['dataset'] == 'CIFAR10':
        out_channel = 10
        in_channel = 3
        g_model_arch = 'CGeneratorA'
        nz = 100
    else:
        raise RuntimeError('Wrong dataset or model_arch parameter setting.')

    model_func = lambda: model_choose_fn.choose_model(config['model_arch'], num_classes=out_channel)
    init_model = model_func()

    ######################################################
    # build up the saving directory
    if not os.path.exists(
            '%sModel/%s/%s_%s_init_mdl.pt' % (conf['savepath'], data_obj.name, conf['dataset'], conf['model_arch'])):
        if not os.path.exists('%sModel/%s/' % (conf['savepath'], data_obj.name)):
            print("Create a new directory")
            os.makedirs('%sModel/%s/' % (conf['savepath'], data_obj.name))

        torch.save(init_model.state_dict(), '%sModel/%s/%s_%s_init_mdl.pt' % (
            conf['savepath'], data_obj.name, conf['dataset'], conf['model_arch']))
    else:
        # Load model
        init_model.load_state_dict(torch.load(
            '%sModel/%s/%s_%s_init_mdl.pt' % (conf['savepath'], data_obj.name, conf['dataset'], conf['model_arch'])))

    ######################################################
    # Begin to train with the specific method
    res_all_performance = []
    if conf['method'] == 'FedFLD':
        print('Train with FedFLD-------------------------')
        g_model_func = lambda: model_choose_fn.choose_g_model(g_model_arch, nz=nz, nc=data_obj.channels,
                                                              img_size=data_obj.width, n_cls=out_channel)
        init_g_model = g_model_func()
        fedavg_res = FedFLD.train_FedFLD(data_obj=data_obj, act_prob=conf['active_frac'],
                                               learning_rate=conf['lr'],
                                               batch_size=conf['bs'], epoch=conf['localE'],
                                               com_amount=conf['comm_amount'],
                                               print_per=conf['print_freq'], weight_decay=conf['weight_decay'],
                                               model_func=model_func,
                                               init_model=init_model, init_g_model=init_g_model,
                                               sch_step=conf['sch_step'], sch_gamma=conf['sch_gamma'],
                                               save_period=conf['save_period'], suffix=config['model_arch'],
                                               trial=False,
                                               data_path=conf['savepath'], rand_seed=conf['seed'],
                                               lr_decay_per_round=conf['lr_decay'],
                                               ewc_lambda=conf['ewc_lambda'],
                                               algo_lambda=conf['his_lambda'])
    else:
        raise RuntimeError('Wrong method.')
    ######################################################

def plotfig(com_amount, res, name, method):
    plt.figure(figsize=(6, 5))
    plt.plot(np.arange(com_amount) + 1, res[:com_amount, 1], label=method)
    plt.ylabel('Test Accuracy', fontsize=16)
    plt.xlabel('Communication Rounds', fontsize=16)
    plt.legend(fontsize=16, loc='lower right', bbox_to_anchor=(1.015, -0.02))
    plt.grid()
    plt.xlim([0, com_amount + 2])
    plt.title(name, fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('%s.pdf' % name, dpi=1000, bbox_inches='tight')


if __name__ == '__main__':
    parser = util_parser.prepare_parser()
    config = vars(parser.parse_args())
    print(config)
    run(config)
