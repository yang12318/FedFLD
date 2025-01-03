from methods.local_fn_gan import *


def train_FedFLD(data_obj, act_prob, learning_rate, batch_size, epoch,
                 com_amount, print_per, weight_decay,
                 model_func, init_model, init_g_model, sch_step, sch_gamma,
                 save_period, suffix='', trial=True, data_path='', 
                 rand_seed=0, lr_decay_per_round=1, ewc_lambda=0.0, his_lambda=0.0):
    suffix = 'FedFLD_' + suffix
    return train_FedGAN_common(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate,
                            batch_size=batch_size, epoch=epoch, com_amount=com_amount,
                            print_per=print_per, weight_decay=weight_decay, model_func=model_func,
                            init_model=init_model, init_g_model=init_g_model, sch_step=sch_step, sch_gamma=sch_gamma,
                            save_period=save_period, suffix=suffix, trial=trial, data_path=data_path, 
                            rand_seed=rand_seed, lr_decay_per_round=lr_decay_per_round, ewc_lambda=ewc_lambda, his_lambda=his_lambda)
