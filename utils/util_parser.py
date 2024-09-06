import argparse

MODEL_ARCH = ['resnet18']
DATASET_NAME = ['CIFAR10']
RULE = ['Dirichlet']
METHODS = ['FedFLD']


def prepare_parser():
    usage = 'Parser for all scripts.'
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--seed',
                        default=1024,
                        type=int,
                        help='random seed')
    parser.add_argument('--model_arch',
                        default="resnet18", choices=MODEL_ARCH,
                        type=str,
                        help='which model architecture is utilized to train')
    parser.add_argument('--method',
                        default='FedFLD', choices=METHODS,
                        type=str,
                        help='which method to be adopted')
    parser.add_argument('--dataset', '-d',
                        default="CIFAR10", choices=DATASET_NAME,
                        type=str,
                        help='which dataset is utilized to train')
    parser.add_argument('--exp_name', '-n',
                        default="Federated",
                        type=str,
                        help='experiment name, used for saving results')
    parser.add_argument('--save', '-s',
                        action='store_true',
                        help='whether save the training results, default is False')
    parser.add_argument('--savepath',
                        default='results/',
                        type=str,
                        help='directory to save exp results')
    parser.add_argument('--print_freq',
                        default=5,
                        type=int,
                        help='print info frequency(ACC) on each client locally')
    parser.add_argument('--save_period',
                        default=1,
                        type=int,
                        help='the frequency of saving the checkpoint')

    parser.add_argument('--n_client',
                        default=10,
                        type=int,
                        help='the number of the clients')
    parser.add_argument('--rule',
                        default="Dirichlet",
                        type=str,
                        choices=RULE,
                        help='split rule of dataset, choices in {iid, Dirichlet}')
    parser.add_argument('--alpha',
                        default=0.01,
                        type=float,
                        help='control the non-iidness of dataset, the parameter of Dirichlet')

    parser.add_argument('--localE',
                        default=5,
                        type=int,
                        help='number of local epochs')
    parser.add_argument('--comm_amount',
                        default=200,
                        type=int,
                        help='number of communication rounds')
    parser.add_argument('--active_frac',
                        default=1.0,
                        type=float,
                        help='the fraction of active clients per communication round')
    parser.add_argument('--bs',
                        default=264,
                        type=int,
                        help='batch size on each client')
    parser.add_argument('--n_minibatch',
                        default=50,
                        type=int,
                        help='the number of minibatch size in SCAFFOLD')
    parser.add_argument('--optimizer',
                        default='SGD',
                        type=str,
                        help='optimizer name')
    parser.add_argument('--lr',
                        default=0.1,
                        type=float,
                        help='client learning rate')
    parser.add_argument('--momentum',
                        default=0.0,
                        type=float,
                        help='local (client) momentum factor')
    parser.add_argument('--weight_decay', '-wd',
                        default=1e-3,
                        type=float,
                        help='local (client) weight decay factor')
    parser.add_argument('--lr_decay', '-ld',
                        default=0.998,
                        type=float,
                        help='local (client) learning rate decay factor')
    
    parser.add_argument('--sch_step',
                        default=1,
                        type=int,
                        help='The learning rate scheduler step')
    parser.add_argument('--sch_gamma',
                        default=1.0,
                        type=float,
                        help='The learning rate scheduler gamma')
    
    parser.add_argument('--ewc_lambda',
                        default=0.7,
                        type=float,
                        help='EWC parameter')
    parser.add_argument('--his_lambda',
                        default=0.5,
                        type=float,
                        help='his parameter')

    return parser
