import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torchmetrics import R2Score, MeanAbsoluteError
from torch import nn


class ModelTrainer:
    def __init__(self, model, loss_fn, optimizer, scheduler, result_name, args, device, params_file=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.params_path = args.params_path
        self.best_val_loss = np.inf
        self.best_epoch = 0
        self.epoch = args.epochs
        self.device = device
        self.result_name = result_name
        self.args = args

        self.__prepare_params_path(params_file)

    def __prepare_params_path(self, params_file):
        if not os.path.exists(self.params_path):
            os.makedirs(self.params_path)
            print(f'Create params directory {self.params_path}', flush=True)

        if not os.path.exists(fr'fig\{self.result_name}'):
            os.makedirs(fr'fig\{self.result_name}', exist_ok=True)

        if params_file:
            params_filename = os.path.join(self.params_path, params_file)
            self.model.load_state_dict(torch.load(params_filename))
            print(f'Load weight from: {params_filename}', flush=True)

    def train_and_validate(self, train_loader, val_loader):
        for epoch in tqdm(range(self.epoch)):
            train_loss, losses = self.__train_epoch(train_loader)
            val_loss = self.__test_epoch(val_loader)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.__save_checkpoint()
            if (epoch + 1) % 10 == 0:
                print(
                    'Epoch: {:03d} | Lr: {:.8f} |Train loss: {:.6f} '
                    '|Val loss: {:.6f}'.format(epoch + 1, self.optimizer.param_groups[0]['lr'],
                                               train_loss,
                                               val_loss))
                for name, loss_value in losses.items():
                    print("{}: {:.3f}".format(name, loss_value))

    def __train_epoch(self, dataloader):
        l_sum, n = 0.0, 0
        self.model.train()
        each_losses = {}
        for x, y in dataloader:
            recon_x, y_pred, hidden = self.model(x)
            l, losses = self.loss_fn(recon_x, x, y_pred, y, hidden)

            for param in self.model.parameters():
                l2_loss = self.args.lambda_l2 * (param.abs() ** 2).sum()

            l += l2_loss

            if 'l2_loss' not in each_losses:
                each_losses['l2_loss'] = 0
            else:
                each_losses['l2_loss'] += l2_loss * x.shape[0]

            for param in self.model.parameters():
                l += self.args.lambda_l2 * (param.abs() ** 2).sum()

            l.backward()
            self.optimizer.step()

            l_sum += l.item() * x.shape[0]
            n += x.shape[0]

            for name, loss_value in losses.items():
                if name not in each_losses:
                    each_losses[name] = 0
                else:
                    each_losses[name] += loss_value * x.shape[0]

        self.scheduler.step()
        loss = l_sum / n
        for name, loss_value in each_losses.items():
            each_losses[name] = loss_value / n

        return loss, each_losses

    def __test_epoch(self, dataloader):
        l_sum, n = 0.0, 0
        self.model.eval()

        for x, y in dataloader:
            recon_x, y_pred, hidden = self.model(x)
            l_rec = torch.nn.L1Loss()(recon_x, x).item()
            l_pred = torch.nn.L1Loss()(y_pred, y).item()

            l_sum += (l_rec + l_pred) * y.shape[0]
            n += y.shape[0]

        return l_sum / n

    def evaluate(self, evaluate_x, evaluate_y):
        x = torch.from_numpy(evaluate_x).float().to(self.device)
        y = torch.from_numpy(evaluate_y).float().to(self.device)

        with torch.no_grad():
            recon_x, y_pred, hidden = self.model(x)
        mae = MeanAbsoluteError().to(self.device)
        r2 = R2Score().to(self.device)
        # recon_x
        recon_res = pd.DataFrame()
        meteo_name = ['Tem', 'U', 'V']

        hidden = hidden.cpu().detach().numpy()
        hidden = pd.DataFrame(hidden).corr()

        print(hidden)
        print(hidden.mean().mean())

        for i in range(recon_x.size(1)):
            for l in range(recon_x.size(2)):
                mae_single = mae(recon_x[:, i, l].reshape(-1), x[:, i, l].reshape(-1)).item()
                recon_res.loc[l, f'mae_{meteo_name[i]}'] = mae_single

                r2_single = r2(recon_x[:, i, l].reshape(-1), x[:, i, l].reshape(-1)).item()
                recon_res.loc[l, f'r2_{meteo_name[i]}'] = r2_single

        mean_values = recon_res.mean()
        mean_row = pd.Series(mean_values, index=recon_res.columns, name="Mean")
        print(mean_row)
        recon_res = pd.concat([recon_res, mean_row.to_frame().T], ignore_index=False)

        # y
        y_dim = self.args.feature
        y_res = pd.DataFrame(columns=y_dim, index=['mae', 'r2'])

        for i in range(y_pred.size(1)):
            mae_single = mae(y_pred[:, i], y[:, i]).item()
            y_res.loc['mae', y_dim[i]] = mae_single

            r2_single = r2(y_pred[:, i], y[:, i]).item()
            y_res.loc['r2', y_dim[i]] = r2_single
        print(y_res)
        if not os.path.exists('result'):
            os.makedirs('result')
        if not os.path.exists(fr'result\{self.result_name}'):
            os.mkdir(fr'result\{self.result_name}')
        y_res.to_csv(os.path.join(fr'result\{self.result_name}', 'pollution_evaluate.csv'))
        recon_res.to_csv(os.path.join(fr'result\{self.result_name}', 'recon_evaluate.csv'))
        hidden.to_csv(os.path.join(fr'result\{self.result_name}', 'corr_evaluate.csv'))
        return recon_res, y_res

    def __save_checkpoint(self):
        params_filename = os.path.join(self.params_path, f'{self.result_name}_params.pth')
        torch.save(self.model.state_dict(), params_filename)
        # print(f'Save parameters to file: {params_filename}', flush=True)

    def result(self, data):
        data = torch.from_numpy(data).float().to(self.device)
        with torch.no_grad():
            recon_x, y_pred, encode_output = self.model(data)
        recon_x = recon_x.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        # Prepare encoded output DataFrame
        hidden = encode_output.cpu().detach().numpy()

        return recon_x, y_pred, hidden
