import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary
from torch.nn import init


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (1, 3, 7), 'kernel size must be 3 or 7'
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 7:
            padding = 3
        elif kernel_size == 1:
            padding = 0

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class PhyAttention(nn.Module):
    def __init__(self):
        super(PhyAttention, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        tem = x[:, 0, :]
        u = x[:, 1, :]
        v = x[:, 2, :]
        temp_diff = torch.diff(tem, dim=-1) + torch.tensor(0.65 * 2, device=tem.device).abs()  # 0.65℃/100m
        temp_att = self.sigmoid(temp_diff)

        wind = torch.sqrt(u ** 2 + v ** 2)
        wind_diff = torch.diff(wind, dim=-1).abs()
        wind_att = self.sigmoid(wind_diff)

        x[:, 0, 1:] = x[:, 0, 1:] * temp_att
        x[:, 1, 1:] = x[:, 1, 1:] * wind_att
        x[:, 2, 1:] = x[:, 2, 1:] * wind_att
        return x, temp_att, wind_att


class ConvBlock(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_channels, out_channels=out_channels, kernel_size=kernel_size,
                              padding=padding, stride=2)

        self.bn = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, in_channels, n_component, hidden_size, attention):
        super().__init__()
        self.attention = attention
        self.spatial_attention = SpatialAttention(1)
        self.bn = nn.BatchNorm1d(in_channels)
        self.conv1 = ConvBlock(in_channels, hidden_size, kernel_size=5, padding=1)
        self.conv2 = ConvBlock(hidden_size, hidden_size, kernel_size=3, padding=0)
        self.conv3 = ConvBlock(hidden_size, hidden_size, kernel_size=3, padding=0)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(hidden_size // 2, n_component, bias=False),
        )

        self.activation = nn.Softmax()

    def forward(self, x):
        if self.attention:
            x = x * self.spatial_attention(x) + x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.activation(x) * 100

        return x


class DeConvActivate(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels=input_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=2, bias=False)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.activation(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, in_channels: int, n_component: int, hidden_size: int):
        super().__init__()
        self.deconv = nn.Sequential(
            DeConvActivate(input_channels=n_component, out_channels=hidden_size, kernel_size=3),
            DeConvActivate(input_channels=hidden_size, out_channels=hidden_size, kernel_size=3),
            DeConvActivate(input_channels=hidden_size, out_channels=hidden_size, kernel_size=3),

            nn.Conv1d(in_channels=hidden_size, out_channels=in_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        x = self.deconv(x)
        return x


class NNet(nn.Module):
    def __init__(self, args):
        super(NNet, self).__init__()
        self.encode = EncoderLayer(in_channels=args.in_channels, n_component=args.n_component,
                                   hidden_size=args.hidden_size, attention=args.attention)
        self.decode = DecoderLayer(in_channels=args.in_channels, n_component=args.n_component,
                                   hidden_size=args.hidden_size)
        self.liner = nn.Linear(args.n_component, len(args.feature))

    def initialize_weights(self):
        for m in self.modules():
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.encode(x)
        hidden = x
        x = x.reshape(x.size(0), x.size(1), 1)
        recon_x = self.decode(x)
        output = self.liner(hidden)
        # output = F.relu(output)
        return recon_x, output, hidden


class GuidedLoss(nn.Module):
    def __init__(self, args):
        super(GuidedLoss, self).__init__()
        self.args = args

        self.recon_criterion = nn.MSELoss()
        self.pollution_criterion = nn.MSELoss()
        self.l1_criterion = nn.L1Loss()

        self.sigmoid = nn.Sigmoid()

    def variance_loss(self, hidden, lambda_factor=0.005):
        mean = hidden.mean(dim=0)
        variance = hidden.var(dim=0)
        target_mean = mean.mean()
        target_std = variance.std()

        kl_loss = 0.5 * torch.sum(
            (mean - target_mean).pow(2) / target_std ** 2 + variance / target_std ** 2 - 1 - torch.log(
                variance / target_std ** 2))
        return kl_loss * lambda_factor

    def feature_independence_loss(self, hidden, lambda_factor):
        mean = torch.mean(hidden, dim=0)
        centered = hidden - mean
        cov = torch.matmul(centered.T, centered) / hidden.shape[0]

        # 计算每列的最小值和最大值
        min_values, _ = cov.min(dim=0)
        max_values, _ = cov.max(dim=0)
        # 归一化操作
        normalized_matrix = (cov - min_values) / (max_values - min_values)
        eye_matrix = torch.eye(normalized_matrix.size(0)).to(cov.device)
        cov_loss = torch.norm(normalized_matrix - eye_matrix, p='fro').pow(2)

        return lambda_factor * cov_loss

    def recon_loss(self, recon_x, x, lambda_factor):
        recon_loss_1 = self.recon_criterion(recon_x[:, 0, :], x[:, 0, :])
        recon_loss_2 = self.recon_criterion(recon_x[:, 1, :], x[:, 1, :])
        recon_loss_3 = self.recon_criterion(recon_x[:, 2, :], x[:, 2, :])

        recon_loss = recon_loss_1 * lambda_factor[0] + recon_loss_2 * lambda_factor[1] + recon_loss_3 * lambda_factor[2]
        return recon_loss

    def physics_loss(self, recon_x, x, lambda_factor):
        temp_diff = torch.diff(x[:, 0, :8], dim=-1) + torch.tensor(0.65 * 2, device=x.device).abs()  # 0.65℃/100m
        temp_diff = self.sigmoid(temp_diff)

        temp_diff_loss = self.l1_criterion(recon_x[:, 0, 1:8] * (1 + temp_diff), x[:, 0, 1:8] * (1 + temp_diff))

        u, v = x[:, 1, :6], x[:, 2, :6]
        wind = torch.sqrt(u ** 2 + v ** 2)
        wind_diff = torch.diff(wind, dim=-1).abs()
        wind_att = self.sigmoid(wind_diff)

        recon_wind = torch.sqrt(recon_x[:, 1, :] ** 2 + recon_x[:, 2, :] ** 2)

        wind_diff_loss = self.l1_criterion(recon_wind[:, 1:6] * (1 + wind_att), wind[:, 1:6] * (1 + wind_att))

        return lambda_factor * temp_diff_loss + lambda_factor * wind_diff_loss

    def forward(self, recon_x, x, pollution_pred, pollution_true, hidden):
        losses = {}

        recon_loss = self.recon_loss(recon_x, x, self.args.reconstruction)
        losses['recon_loss'] = recon_loss.item()

        pollution_loss = self.args.regression * self.pollution_criterion(pollution_pred, pollution_true)
        losses['pollution_loss'] = pollution_loss.item()

        total_loss = recon_loss + pollution_loss

        if self.args.physics > 0:
            physics_loss = self.physics_loss(recon_x, x, self.args.physics)
            total_loss += physics_loss
            losses['physics_loss'] = physics_loss.item()

        if self.args.independence > 0:
            feature_independence_loss = self.feature_independence_loss(hidden, self.args.independence)
            total_loss += feature_independence_loss
            losses['feature_independence_loss'] = feature_independence_loss.item()

        if self.args.variance > 0:
            variance_loss = self.variance_loss(hidden, self.args.variance)
            total_loss += variance_loss
            losses['variance_loss'] = variance_loss.item()

        return total_loss, losses


if __name__ == '__main__':
    model = NNet(in_channels=4, n_component=6, hidden_size=64, output_size=6)
    x = torch.randn((128, 4, 15))
    recon_x, y_pred, hidden = model(x)
    print(recon_x.size(), y_pred.size())
    summary(model, (128, 4, 15))
