import torch
import torch.nn as nn
import torch.autograd.profiler as profiler


class TopKEst(nn.Module):
    def __init__(self, ts_size=320, dis_encoding_size=5, ts_embedding_size=80, K=30):
        super(TopKEst, self).__init__()
        self.ts_size = ts_size
        self.dis_encoding_size = dis_encoding_size
        self.ts_embedding_size = ts_embedding_size
        self.K = K
        self.weight_hidden_layer_sizes = [128, 128, 64]
        self.vae_hidden_units = [512, 512, 256]
        self.vae_loss = nn.MSELoss()

        self.dis_net = nn.Sequential(
            nn.Linear(1, self.dis_encoding_size * 2),
            nn.ELU(),
            nn.Linear(self.dis_encoding_size * 2, self.dis_encoding_size),
            nn.ELU()
        )

        self.weight_net = nn.Sequential(
            nn.Linear(self.dis_encoding_size*2 + self.ts_embedding_size * 2, self.weight_hidden_layer_sizes[0]),
            nn.ELU(),
            nn.Linear(self.weight_hidden_layer_sizes[0], self.weight_hidden_layer_sizes[1]),
            nn.ELU(),
            nn.Linear(self.weight_hidden_layer_sizes[1], self.weight_hidden_layer_sizes[2]),
            # nn.Dropout(),
            nn.ELU(),
            nn.Linear(self.weight_hidden_layer_sizes[2], 1),
            nn.ReLU()
        )


        self.encoder_net = nn.Sequential(
            nn.Linear(self.ts_size, self.vae_hidden_units[0]),
            nn.ELU(),
            nn.Linear(self.vae_hidden_units[0], self.vae_hidden_units[1]),
            nn.Dropout(),
            nn.ELU(),
            nn.Linear(self.vae_hidden_units[1], self.vae_hidden_units[2]),
            nn.ELU(),
            nn.Linear(self.vae_hidden_units[2], self.ts_embedding_size),
            # nn.ELU()
        )

        self.decoder_net = nn.Sequential(
            nn.Linear(self.ts_embedding_size, self.vae_hidden_units[2]),
            nn.ELU(),
            nn.Linear(self.vae_hidden_units[2], self.vae_hidden_units[1]),
            nn.Dropout(),
            nn.ELU(),
            nn.Linear(self.vae_hidden_units[1], self.vae_hidden_units[0]),
            nn.ELU(),
            nn.Linear(self.vae_hidden_units[0], self.ts_size),
            # nn.ReLU()
        )


        self.mse_loss = torch.nn.MSELoss()
        self.huber_loss = torch.nn.HuberLoss(delta=1.345)


    def ae_forward(self, tss):
        z = self.encoder_net(tss)
        zf = z
        x = self.decoder_net(z)

        e = torch.abs(x - tss)
        # loss = self.vae_loss(x, tss)
        loss = torch.mean(torch.sum(e, axis=1))
        return zf, loss


    def self_huber_loss(self, preds, labels):
        diff = torch.abs(preds - labels)
        mask = (diff <= 2.345).float()

        loss = 0.5 * diff ** 2
        loss = loss * mask + (0.345 * diff - 0.5 * 0.345 ** 2) * (1 - mask)

        return loss.mean()



    def return_weight(self, target_ts, true_cards, thresholds, topk_tss, topk_dists, topk_cards):
        target_em, loss_ae = self.ae_forward(target_ts)
        thresholds_ec = self.dis_net(thresholds)

        topk_em, _loss = self.ae_forward(topk_tss)
        ds_ec = self.dis_net(topk_dists)

        thresholds_ec = thresholds_ec.unsqueeze(1).repeat(1, self.K, 1)
        dist_c = torch.concat((thresholds_ec, ds_ec), dim=2)
        target_em = target_em.unsqueeze(1).repeat(1, self.K, 1)

        tss_em_c = torch.concat((target_em, topk_em), dim=2)
        em = torch.concat((tss_em_c, dist_c), dim=2)
        weight = self.weight_net(em)
        return weight


    def forward(self, target_ts, true_cards, thresholds, topk_tss, topk_dists, topk_cards):
        target_em, loss_ae = self.ae_forward(target_ts)

        loss_ae *= 0.01
        thresholds_ec = self.dis_net(thresholds)

        topk_em, _loss = self.ae_forward(topk_tss)


        loss_ae += 0.01*_loss
        ds_ec = self.dis_net(topk_dists)

        # start_time2 = time.time()
        thresholds_ec = thresholds_ec.unsqueeze(1).repeat(1, self.K, 1)
        dist_c = torch.concat((thresholds_ec, ds_ec), dim=2)
        target_em = target_em.unsqueeze(1).repeat(1, self.K, 1)
        tss_em_c = torch.concat((target_em, topk_em), dim=2)
        em = torch.concat((tss_em_c, dist_c), dim=2)


        weight = self.weight_net(em)
        est_cards = weight*topk_cards
        est_cards = torch.sum(est_cards, dim=1)
        loss = self.self_huber_loss(torch.log_(est_cards+1), torch.log_(true_cards))

        return est_cards, loss_ae, loss