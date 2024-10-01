import math
import time

import torch.nn
import torch
import numpy as np
import TopKEst
import utility_functions
import obtain_batch


import sys

device = 'cuda'
if torch.cuda.is_available():
    device = 'cuda'
    print('cuda used')
else:
    print('cpu used')
    device = 'cpu'


dataset_name = sys.argv[1]
model_path = sys.argv[2]
TS_SIZE = int(sys.argv[3])
torch.cuda.set_device(int(sys.argv[4]))

prefix_dir = f'../data/{dataset_name}'
SAMPLE_SIZE = 30

def train():
    tr_tss, tr_ths, tr_cards, _ = utility_functions.load_labeled_data(TS_SIZE,
                                                                              f'{prefix_dir}/{dataset_name}_trainingData.npy',
                                                                              False, False)
    tr_kcards, tr_ktss, tr_kdists = utility_functions.gen_training_data2(prefix_dir, dataset_name, 'training')
    tr_tss_vae = np.load(f'{prefix_dir}/{dataset_name}_training_tss.npy')
    tr_kcards = tr_kcards.reshape((tr_kcards.shape[0], tr_kcards.shape[1], 1))

    te_tss, te_ths, te_cards, _ = utility_functions.load_labeled_data(TS_SIZE,
                                                                                                      f'{prefix_dir}/{dataset_name}_testingData.npy',
                                                                                                      False, False)
    te_kcards, te_ktss, te_kdists = utility_functions.gen_training_data2(prefix_dir, dataset_name,'testing')
    te_kcards = te_kcards.reshape((te_kcards.shape[0], te_kcards.shape[1], 1))


    start_t = time.time()
    TopkEstM = TopKEst.TopKEst(TS_SIZE, 5, 80, SAMPLE_SIZE)
    weight_decay = 0.0001
    betas = (0.9, 0.999)
    batch_size = 64
    optimizer_vae = torch.optim.Adam(TopkEstM.parameters(), 0.0001, weight_decay=weight_decay, betas=betas)
    optimizer_est = torch.optim.Adam(TopkEstM.parameters(), 0.002, weight_decay=weight_decay, betas=betas)


    TopkEstM.to(device=device)

    for epoch in range(30):
        print("vae epoch:{0}".format(epoch))
        if epoch % 5 == 0 and epoch > 0:
            for p in optimizer_est.param_groups:
                p['lr'] *= 0.7
        TopkEstM.train()
        n_batch = int(tr_tss_vae.shape[0] / batch_size) + 1
        total_loss = 0
        for b in range(n_batch):
            _tr_ts_vae = obtain_batch.get_batch_vae(b, batch_size, tr_tss_vae)
            _tr_ts_vae = _tr_ts_vae.to(device=device)
            pred, loss = TopkEstM.ae_forward(_tr_ts_vae)
            total_loss += loss
            optimizer_vae.zero_grad()
            loss.backward()
            optimizer_vae.step()
        print(f'Loss/Train VAE {total_loss / n_batch} {epoch}')

    current_best_q = 99999999.0
    for epoch in range(30):
        print("est epoch:{0}".format(epoch))
        if epoch % 5 == 0 and epoch > 0:
            for p in optimizer_est.param_groups:
                p['lr'] *= 0.5
        TopkEstM.train()
        n_batch = int(tr_cards.shape[0] / batch_size) + 1
        t_loss = 0
        for b in range(n_batch):
            _tr_ts, _tr_card, _tr_th, _tr_kts, _tr_kdist, _tr_kcard = obtain_batch.get_batch_est(SAMPLE_SIZE, epoch, b, batch_size, tr_tss, tr_cards, tr_ths, tr_ktss, tr_kdists, tr_kcards)



            _tr_kcard = _tr_kcard.to(device=device)
            _tr_kdist = _tr_kdist.to(device=device)
            _tr_kts = _tr_kts.to(device=device)

            _tr_card = _tr_card.to(device=device)
            _tr_ts = _tr_ts.to(device=device)
            _tr_th = _tr_th.to(device=device)


            pred, loss_ae, loss = TopkEstM(_tr_ts, _tr_card, _tr_th, _tr_kts, _tr_kdist, _tr_kcard)
            if b % 500 == 0:
                print(f'epoch: {epoch}, batch: {b}, loss: {loss}')

            t_loss += loss

            optimizer_est.zero_grad()
            loss += loss_ae
            loss.backward()

            optimizer_est.step()


        print(f'Loss/train Est {t_loss / n_batch} {epoch}')

        n_batch = int(te_cards.shape[0] / batch_size) + 1
        TopkEstM.eval()
        with torch.no_grad():
            predictions = None
            for bi in range(n_batch):
                _te_ts, _te_card, _te_th, _te_kts, _te_kdist, _te_kcard = obtain_batch.get_batch_est(SAMPLE_SIZE, 0, bi, batch_size,
                                                                                                     te_tss, te_cards,
                                                                                                     te_ths,
                                                                                                     te_ktss, te_kdists,
                                                                                                     te_kcards, False)

                _te_kcard = _te_kcard.to(device=device)
                _te_kdist = _te_kdist.to(device=device)
                _te_kts = _te_kts.to(device=device)
                _te_card = _te_card.to(device=device)
                _te_ts = _te_ts.to(device=device)
                _te_th = _te_th.to(device=device)

                pred, loss_ae, loss = TopkEstM(_te_ts, _te_card, _te_th, _te_kts, _te_kdist, _te_kcard)

                pred = pred.cpu().numpy()
                if bi == 0:
                    predictions = pred
                else:
                    predictions = np.concatenate((predictions, pred), axis=0)
            predictions = predictions[:te_cards.shape[0]]
            e = np.abs(predictions - te_cards)
            # s = np.mean(np.sqrt(np.sum(e * e, axis=1)))
            s = np.mean(np.sum(e, axis=1))
            # s = np.max(np.sum(e, axis=1))
            print(f'Loss/test Validation {s}')
            e = eval(predictions, te_cards)
            if e[3] < current_best_q:
                current_best_q = e[3]
                torch.save(TopkEstM.state_dict(), model_path)
            torch.save(TopkEstM.state_dict(), f'{model_path}_latest')
        # print(f'Loss/train Validation {t_loss / n_batch} {epoch}')

    end_t = time.time()
    print(end_t - start_t)

def inference():
    TopkEstM = TopKEst.TopKEst(TS_SIZE, 5, 80, SAMPLE_SIZE)
    TopkEstM.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # model 4: ~28
    batch_size = 640

    te_tss, te_ths, te_cards, _ = utility_functions.load_labeled_data(TS_SIZE,
                                                              f'{prefix_dir}/{dataset_name}_testingData.npy',
                                                              False, False)
    te_kcards, te_ktss, te_kdists = utility_functions.gen_training_data2(prefix_dir, dataset_name, 'testing')
    te_kcards = te_kcards.reshape((te_kcards.shape[0], te_kcards.shape[1], 1))

    n_batch = int(te_ktss.shape[0] / batch_size) + 1

    TopkEstM.eval()

    with torch.no_grad():
        predictions = None
        s = time.time()
        for b in range(n_batch):
            _te_ts, _te_card, _te_th, _te_kts, _te_kdist, _te_kcard = obtain_batch.get_batch_est(SAMPLE_SIZE, 0, b, batch_size,
                                                                                                     te_tss, te_cards,
                                                                                                     te_ths,
                                                                                                     te_ktss, te_kdists,
                                                                                                     te_kcards, False)

            pred, loss_ae, loss = TopkEstM(_te_ts, _te_card, _te_th, _te_kts, _te_kdist, _te_kcard)
            pred = pred.cpu().numpy()
            pred = pred.reshape((pred.shape[0],pred.shape[1]))
            if b == 0:
                predictions = pred
            else:
                predictions = np.concatenate((predictions, pred))
                #predictions = np.concatenate((predictions, pred), axis=0)
        predictions = predictions[:te_cards.shape[0]]
        e = time.time()
        print(e-s)
        print(predictions.shape, te_cards.shape)
        utility_functions.eval(np.squeeze(predictions), np.squeeze(te_cards))

train()
inference()