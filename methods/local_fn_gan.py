import copy
import torch
from torch.utils import data
from utils.util_dataset import Dataset
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from models import model_choose_fn
from methods.train_gan_fn import *

# Global parameters
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_norm = 10
train_fedgen_feature = True

# --- Evaluate a NN model
def get_acc_loss(data_x, data_y, model, dataset_name, w_decay=None):
    acc_overall = 0
    loss_overall = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    batch_size = min(2000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_gen = data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=batch_size, shuffle=False)
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / batch_size))):
            batch_x, batch_y = tst_gen_iter.__next__()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            y_pred = model(batch_x)

            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss_overall += loss.item()

            # Accuracy calculation
            y_pred = y_pred.cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct

    loss_overall /= n_tst
    if w_decay is not None:
        # Add L2 loss
        params = get_mdl_params([model], n_par=None)
        loss_overall += w_decay / 2 * np.sum(params * params)

    model.train()
    return loss_overall, acc_overall / n_tst


# --- Helper functions

def avg_models(mdl, clnt_models, weight_list):
    n_node = len(clnt_models)
    dict_list = list(range(n_node))
    for i in range(n_node):
        dict_list[i] = copy.deepcopy(dict(clnt_models[i].state_dict()))

    param_0 = clnt_models[0].state_dict()

    for name, param in param_0:
        param_ = weight_list[0] * param.data
        for i in list(range(1, n_node)):
            param_ = param_ + weight_list[i] * dict_list[i][name].data
        dict_list[0][name].data.copy_(param_)

    mdl.load_state_dict(dict_list[0])

    # Remove dict_list from memory
    del dict_list

    return mdl


def train_model(model, trn_x, trn_y, tst_x, tst_y, learning_rate, batch_size, epoch, print_per, weight_decay,
                dataset_name, sch_step=1, sch_gamma=1):
    n_trn = trn_x.shape[0]
    trn_gen = data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name), batch_size=batch_size,
                              shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train()
    model = model.to(device)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)

    # Put tst_x=False if no tst data given
    print_test = not isinstance(tst_x, bool)

    loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
    if print_test:
        loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
        print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
              % (0, acc_trn, loss_trn, acc_tst, loss_tst, optimizer.param_groups[0]['lr']))
    else:
        print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f"
              % (0, acc_trn, loss_trn, optimizer.param_groups[0]['lr']))

    model.train()

    for e in range(epoch):
        # Training

        trn_gen_iter = trn_gen.__iter__()
        for i in range(int(np.ceil(n_trn / batch_size))):
            batch_x, batch_y = trn_gen_iter.__next__()

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())

            loss = loss / list(batch_y.size())[0]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                           max_norm=max_norm)  # Clip gradients to prevent exploding
            optimizer.step()

        if (e + 1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
            if print_test:
                loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f, LR: %.4f"
                      % (e + 1, acc_trn, loss_trn, acc_tst, loss_tst, optimizer.param_groups[0]['lr']))
            else:
                print("Epoch %3d, Training Accuracy: %.4f, Loss: %.4f, LR: %.4f" % (
                    e + 1, acc_trn, loss_trn, optimizer.param_groups[0]['lr']))

            model.train()
        scheduler.step()

    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()

    return model

def train_global_model_cgan_bs_weight_iloop_fz_dis(s_model, g_model, t_model, client_params,
                                      clnt_cls_num, glb_lr, gen_lr,
                                      batch_size, print_per, weight_decay,
                                      dataset_name, trn_x, trn_y, tst_x, tst_y, i, suffix, data_obj, data_path,
                                      ewc_lambda, hislambda, hist_gan_diffs):
    t_model.eval()
    s_model.to(device)
    t_model.to(device)
    g_model.to(device)

    init_model = g_model.to(device)

    num_clients, num_classes = clnt_cls_num.shape
    optimizer_D = torch.optim.SGD(s_model.parameters(), lr=glb_lr, weight_decay=weight_decay)
    optimizer_G = torch.optim.Adam(g_model.parameters(), lr=gen_lr)
    iterations = 10
    inner_round_g = 1
    inner_round_d = 5
    nz = 100

    for params in s_model.parameters():
        params.requires_grad = True

    cls_num = np.sum(clnt_cls_num, axis=0)
    cls_clnt_weight = clnt_cls_num / (np.tile(cls_num[np.newaxis, :], (num_clients, 1)) + 1e-6)
    cls_clnt_weight = cls_clnt_weight.transpose()
    labels_all = generate_labels(iterations * batch_size, cls_num)

    for e in range(iterations):

        labels = labels_all[e*batch_size:(e*batch_size+batch_size)]
        batch_weight = torch.Tensor(get_batch_weight(labels, cls_clnt_weight)).cuda()
        onehot = np.zeros((batch_size, num_classes))
        onehot[np.arange(batch_size), labels] = 1
        y_onehot = torch.Tensor(onehot).cuda()
        z = torch.randn((batch_size, nz, 1, 1)).cuda()

        ############## train generator ##############
        s_model.eval()
        g_model.train()
        loss_G = 0
        loss_tran_total = 0
        loss_cls_total = 0
        loss_div_total = 0
        ewc_value_total=0
        for _ in range(inner_round_g):
            for client in range(num_clients):
                optimizer_G.zero_grad()
                t_model = set_client_from_params(t_model, client_params[client])
                loss, loss_tran, loss_cls, loss_div, ewc_value = train_generator(z, y_onehot, labels, g_model,
                                                                                     s_model, t_model,
                                                                                     batch_weight[:, client], num_clients,
                                                                                     i, suffix, data_obj, data_path, ewc_lambda)
                loss_G += loss
                loss_tran_total += loss_tran
                loss_cls_total += loss_cls
                loss_div_total += loss_div
                ewc_value_total += ewc_value
                optimizer_G.step()
        
        if i > 1:
            g_model_pre = init_model
            g_model_pre.load_state_dict(torch.load('%sModel/%s/%s/%dcom_G.pt'
                                                       % (data_path, data_obj.name, suffix, i-1)))
            idx_nonbn_gan = get_mdl_nonbn_idx([init_model])[0]
            n_par_gan = len(get_mdl_params([init_model])[0])
            gan_pre_param = get_mdl_params([g_model_pre], n_par_gan)[0]

            g_param=get_mdl_params([g_model], n_par_gan)[0]
            hist_gan_diffs += g_param - gan_pre_param
            g_param[idx_nonbn_gan] = g_param[idx_nonbn_gan] + hist_gan_diffs[idx_nonbn_gan]
            g_model_his = init_model
            g_model_his = set_client_from_params(g_model_his, g_param)

        ############## train student model ##############
        s_model.train()
        g_model.eval()

        if i > 1:
            s_model.train()
            g_model.eval()

            for _ in range(inner_round_d):
                optimizer_D.zero_grad()
                fake = g_model(z, y_onehot).detach()
                fake_his = g_model_his(z, y_onehot).detach()
                s_logit = s_model(fake)
                s_logit_his = s_model(fake_his)
                t_logit_merge = 0
                t_logit_merge_his = 0
                for client in range(num_clients):
                    t_model = set_client_from_params(t_model, client_params[client])
                    t_logit = t_model(fake).detach()
                    t_logit_his = t_model(fake_his).detach()
                    t_logit_merge += F.softmax(t_logit, dim=1) * batch_weight[:, client][:, np.newaxis].repeat(1, num_classes)
                    t_logit_merge_his += F.softmax(t_logit_his, dim=1) * batch_weight[:, client][:, np.newaxis].repeat(1, num_classes)
                loss_kd = torch.mean(-F.log_softmax(s_logit, dim=1) * t_logit_merge) + his_lambda * torch.mean(-F.log_softmax(s_logit_his, dim=1) * t_logit_merge_his)
                loss_kd.backward()
                optimizer_D.step()
        else:
            for _ in range(inner_round_d):
                optimizer_D.zero_grad()
                fake = g_model(z, y_onehot).detach()
                s_logit = s_model(fake)
                t_logit_merge = 0
                for client in range(num_clients):
                    t_model = set_client_from_params(t_model, client_params[client])
                    t_logit = t_model(fake).detach()
                    t_logit_merge += F.softmax(t_logit, dim=1) * batch_weight[:, client][:, np.newaxis].repeat(1, num_classes)

                loss_ks = torch.mean(-F.log_softmax(s_logit, dim=1) * t_logit_merge)
                loss_ks.backward()
                optimizer_D.step()

        if (e + 1) % print_per == 0:
            loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, s_model, dataset_name)
            loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, s_model, dataset_name)
            s_model.train()

    # Freeze model
    for params in s_model.parameters():
        params.requires_grad = False
    s_model.eval()

    return s_model, hist_gan_diffs

def set_client_from_params(mdl, params):
    dict_param = copy.deepcopy(dict(mdl.state_dict()))
    idx = 0
    for name, param in dict(mdl.state_dict()).items():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(torch.tensor(params[idx:idx + length].reshape(weights.shape)).to(device))
        idx += length

    mdl.load_state_dict(dict_param)
    return mdl


def get_mdl_params(model_list, n_par=None):
    if n_par is None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in dict(exp_mdl.state_dict()).items():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in dict(mdl.state_dict()).items():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)

def get_mdl_nonbn_idx(model_list, name_par=None):
    if name_par is None:
        exp_mdl = model_list[0]
        name_par = []
        for name, param in exp_mdl.named_parameters():
            name_par.append(name)

    idx_list = [[]] * len(model_list)
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in dict(mdl.state_dict()).items():
            temp = param.data.cpu().numpy().reshape(-1)
            if name in name_par:
                idx_list[i].extend(list(range(idx, idx + len(temp))))
            idx += len(temp)
    return np.array(idx_list)

def train_FedGAN_common(data_obj, act_prob, learning_rate, batch_size, epoch,
                     com_amount, print_per, weight_decay,
                     model_func, init_model, init_g_model, sch_step, sch_gamma,
                     save_period, suffix='', trial=True, data_path='', rand_seed=0, lr_decay_per_round=1,
                     ewc_lambda=1.0, his_lambda=0.0):
    glb_model_lr = 0.1
    gen_model_lr = 0.01
    init_g_model = init_g_model.to(device)
    clnt_cls_num = get_class_number(data_obj)

    suffix += '_S%d_F%f_Lr%f_%d_%f_B%d_E%d_W%f_e%f_h%f' % (
        save_period, act_prob, learning_rate, sch_step, sch_gamma, batch_size, epoch, weight_decay, ewc_lambda, his_lambda)

    suffix += '_lrdecay%f' % lr_decay_per_round
    suffix += '_seed%d' % rand_seed

    n_clnt = data_obj.n_client

    clnt_x = data_obj.clnt_x
    clnt_y = data_obj.clnt_y

    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)

    weight_list = np.asarray([len(clnt_y[i]) for i in range(n_clnt)])
    weight_list = weight_list.reshape((n_clnt, 1))

    if (not trial) and (not os.path.exists('%sModel/%s/%s' % (data_path, data_obj.name, suffix))):
        os.mkdir('%sModel/%s/%s' % (data_path, data_obj.name, suffix))

    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances))
    fed_mdls_all = list(range(n_save_instances))
    fed_mdls_ft = list(range(n_save_instances))

    trn_perf_sel = np.zeros((com_amount, 2))
    trn_perf_all = np.zeros((com_amount, 2))
    trn_perf_ft = np.zeros((com_amount, 2))

    tst_perf_sel = np.zeros((com_amount, 2))
    tst_perf_all = np.zeros((com_amount, 2))
    tst_perf_ft = np.zeros((com_amount, 2))

    n_par = len(get_mdl_params([model_func()])[0])
    idx_nonbn = get_mdl_nonbn_idx([model_func()])[0]

    init_par_list = get_mdl_params([init_model], n_par)[0]
    clnt_params_list = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1)  # n_clnt X n_par
    hist_avg_diffs = np.zeros((n_clnt, n_par)).astype('float32')

    n_par_gan = len(get_mdl_params([init_g_model])[0])

    hist_gan_diffs = np.zeros((n_par_gan,)).astype('float32')

    saved_itr = -1
    # writer object is for tensorboard visualization, comment out if not needed
    writer = SummaryWriter('%sRuns/%s/%s' % (data_path, data_obj.name, suffix))

    if not trial:
        # Check if there are past saved iterates
        for i in range(com_amount):
            if os.path.exists('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, i + 1)):
                saved_itr = i

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                                                     % (data_path, data_obj.name, suffix, i + 1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr // save_period] = fed_model

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_all.pt'
                                                     % (data_path, data_obj.name, suffix, i + 1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_all[saved_itr // save_period] = fed_model

                if os.path.exists('%sModel/%s/%s/%dcom_trn_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1))):
                    trn_perf_sel[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_trn_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    trn_perf_all[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_trn_perf_all.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    trn_perf_ft[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_trn_perf_ft.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    tst_perf_sel[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    tst_perf_all[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_tst_perf_all.npy' % (data_path, data_obj.name, suffix, (i + 1)))
                    tst_perf_ft[:i + 1] = np.load(
                        '%sModel/%s/%s/%dcom_tst_perf_ft.npy' % (data_path, data_obj.name, suffix, (i + 1)))

                    clnt_params_list = np.load(
                        '%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, i + 1))
                    
                    if i > 2:
                        hist_avg_diffs = np.load(
                            '%sModel/%s/%s/%d_hist_avg_diffs.npy' % (data_path, data_obj.name, suffix, i))
                         hist_gan_diffs = np.load(
                            '%sModel/%s/%s/%d_hist_gan_diffs.npy' % (data_path, data_obj.name, suffix, i))

    if trial or (not os.path.exists('%sModel/%s/%s/%dcom_sel.pt' % (data_path, data_obj.name, suffix, com_amount))):
        # clnt_models = list(range(n_clnt))
        if saved_itr == -1:
            avg_model = model_func().to(device)
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.state_dict())))

            all_model = model_func().to(device)
            all_model.load_state_dict(copy.deepcopy(dict(init_model.state_dict())))

            avg_model_ft = model_func().to(device)
            avg_model_ft.load_state_dict(copy.deepcopy(dict(init_model.state_dict())))

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(init_model.state_dict())))

            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]
            ft_mdl_param = get_mdl_params([avg_model_ft], n_par)[0]

        else:
            # Load recent one
            avg_model = model_func().to(device)
            avg_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_sel.pt'
                                                 % (data_path, data_obj.name, suffix, (saved_itr + 1))))

            all_model = model_func().to(device)
            all_model.load_state_dict(torch.load('%sModel/%s/%s/%dcom_all.pt'
                                                 % (data_path, data_obj.name, suffix, (saved_itr + 1))))

            avg_model_ft = model_func().to(device)
            avg_model_ft.load_state_dict(torch.load('%sModel/%s/%s/%dcom_ft.pt'
                                                 % (data_path, data_obj.name, suffix, (saved_itr + 1))))
            ft_mdl_param = get_mdl_params([avg_model_ft], n_par)[0]

            cur_cld_model = model_func().to(device)
            cur_cld_model.load_state_dict(copy.deepcopy(dict(fed_model.state_dict())))
            cld_mdl_param = get_mdl_params([cur_cld_model], n_par)[0]

        for i in range(saved_itr + 1, com_amount):
            # Train if doesn't exist
            # Fix randomness
            inc_seed = 0
            while True:
                np.random.seed(i + rand_seed + inc_seed)
                act_list = np.random.uniform(size=n_clnt)
                act_clients = act_list <= act_prob
                selected_clnts = np.sort(np.where(act_clients)[0])
                inc_seed += 1
                # Choose at least one client in each synch
                if len(selected_clnts) != 0:
                    break

            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clnts])))

            # del clnt_models
            clnt_models = list(range(n_clnt))
            for clnt in selected_clnts:
                print('---- Training client %d' % clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]
                tst_x = False
                tst_y = False

                clnt_models[clnt] = model_func().to(device)

                clnt_models[clnt].load_state_dict(copy.deepcopy(dict(avg_model_ft.state_dict())))

                for params in clnt_models[clnt].parameters():
                    params.requires_grad = True
                clnt_models[clnt] = train_model(clnt_models[clnt], trn_x, trn_y,
                                                tst_x, tst_y,
                                                learning_rate * (lr_decay_per_round ** i), batch_size, epoch, print_per,
                                                weight_decay,
                                                data_obj.dataset, sch_step, sch_gamma)

                curr_model_par = get_mdl_params([clnt_models[clnt]], n_par)[0]
                hist_avg_diffs[clnt] += curr_model_par - ft_mdl_param
                clnt_params_list[clnt] = curr_model_par

            # Scale with weights

            avg_mdl_param_sel = np.mean(clnt_params_list[selected_clnts], axis=0)

            cld_mdl_param = avg_mdl_param_sel.copy()
            cld_mdl_param[idx_nonbn] = cld_mdl_param[idx_nonbn] + np.mean(hist_avg_diffs, axis=0)[idx_nonbn]#更新

            avg_model = set_client_from_params(model_func(), np.sum(
                clnt_params_list[selected_clnts] * weight_list[selected_clnts] / np.sum(weight_list[selected_clnts]),
                axis=0))
            all_model = set_client_from_params(model_func(),
                                               np.sum(clnt_params_list * weight_list / np.sum(weight_list), axis=0))

            trn_x_sel = np.concatenate(clnt_x[selected_clnts], axis=0)
            trn_y_sel = np.concatenate(clnt_y[selected_clnts], axis=0)
            avg_model_ft = set_client_from_params(model_func().to(device), cld_mdl_param)
            avg_model_ft, hist_gan_diffs = train_global_model_cgan_bs_weight_iloop_fz_dis(avg_model_ft, init_g_model, model_func(), clnt_params_list[selected_clnts], 
                                              clnt_cls_num[selected_clnts], glb_model_lr * (lr_decay_per_round ** i), gen_model_lr * (lr_decay_per_round ** i),
                                              batch_size, print_per, weight_decay, data_obj.dataset,
                                              trn_x_sel, trn_y_sel, data_obj.tst_x, data_obj.tst_y, i, suffix, data_obj, data_path, ewc_lambda, his_lambda, hist_gan_diffs)

            ft_mdl_param = get_mdl_params([avg_model_ft], n_par)[0]
            loss_tst, acc_tst = get_acc_loss(data_obj.tst_x, data_obj.tst_y,
                                            avg_model, data_obj.dataset, 0)
            tst_perf_sel[i] = [loss_tst, acc_tst]

            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if (not trial) and ((i + 1) % save_period == 0):
                torch.save(avg_model.state_dict(), '%sModel/%s/%s/%dcom_sel.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))
                torch.save(all_model.state_dict(), '%sModel/%s/%s/%dcom_all.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))
                torch.save(avg_model_ft.state_dict(), '%sModel/%s/%s/%dcom_ft.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                torch.save(init_g_model.state_dict(), '%sModel/%s/%s/%dcom_G.pt'
                           % (data_path, data_obj.name, suffix, (i + 1)))

                np.save('%sModel/%s/%s/%dcom_trn_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        trn_perf_sel[:i + 1])
                np.save('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_perf_sel[:i + 1])

                np.save('%sModel/%s/%s/%dcom_trn_perf_all.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        trn_perf_all[:i + 1])
                np.save('%sModel/%s/%s/%dcom_tst_perf_all.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_perf_all[:i + 1])

                np.save('%sModel/%s/%s/%dcom_trn_perf_ft.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        trn_perf_ft[:i + 1])
                np.save('%sModel/%s/%s/%dcom_tst_perf_ft.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        tst_perf_ft[:i + 1])

                np.save('%sModel/%s/%s/%d_clnt_params_list.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        clnt_params_list)
                if i > 1:
                    np.save('%sModel/%s/%s/%d_hist_avg_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        hist_avg_diffs)
                    np.save('%sModel/%s/%s/%d_hist_gan_diffs.npy' % (data_path, data_obj.name, suffix, (i + 1)),
                        hist_gan_diffs)

                if (i + 1) > save_period:
                    if os.path.exists('%sModel/%s/%s/%dcom_trn_perf_sel.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period)):
                        # Delete the previous saved arrays
                        os.remove('%sModel/%s/%s/%dcom_trn_perf_sel.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_sel.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))

                        os.remove('%sModel/%s/%s/%dcom_trn_perf_all.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_all.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))

                        os.remove('%sModel/%s/%s/%dcom_trn_perf_ft.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))
                        os.remove('%sModel/%s/%s/%dcom_tst_perf_ft.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))

                        os.remove('%sModel/%s/%s/%d_clnt_params_list.npy' % (
                            data_path, data_obj.name, suffix, i + 1 - save_period))
                        
                        if i > 3:
                            os.remove('%sModel/%s/%s/%dcom_sel.pt'
                                    % (data_path, data_obj.name, suffix, i- save_period))
                            os.remove('%sModel/%s/%s/%dcom_all.pt'
                                    % (data_path, data_obj.name, suffix, i- save_period))
                            os.remove('%sModel/%s/%s/%dcom_ft.pt'
                                    % (data_path, data_obj.name, suffix, i- save_period))
                            os.remove('%sModel/%s/%s/%dcom_G.pt'
                                    % (data_path, data_obj.name, suffix, i- save_period))
                            os.remove('%sModel/%s/%s/%d_hist_avg_diffs.npy' % (
                            data_path, data_obj.name, suffix, i- save_period))
                            os.remove('%sModel/%s/%s/%d_hist_gan_diffs.npy' % (
                            data_path, data_obj.name, suffix, i- save_period))

            if ((i + 1) % save_period == 0):
                fed_mdls_sel[i // save_period] = avg_model
                fed_mdls_all[i // save_period] = all_model
                fed_mdls_ft[i // save_period] = avg_model_ft

            torch.cuda.empty_cache()

    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, \
           fed_mdls_all, trn_perf_all, tst_perf_all, \
           fed_mdls_ft, trn_perf_ft, tst_perf_ft
