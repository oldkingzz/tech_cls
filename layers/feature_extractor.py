import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    残差块，用于深层特征提取
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels)
        )
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        residual = self.shortcut(residual)
        
        out += residual
        out = self.relu(out)
        
        return out

class SEModule(nn.Module):
    """
    Squeeze-and-Excitation模块，增强特征通道间的依赖关系
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        batch_size, channels, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.view(batch_size, channels, 1)
        return x * y

class FFTFeatureLayer(nn.Module):
    """
    频域特征提取层，使用FFT分析频率特征
    """
    def __init__(self, d_model, freq_bins=128):
        super().__init__()
        self.freq_bins = freq_bins
        self.fft_projection = nn.Sequential(
            nn.Linear(freq_bins, d_model),
            nn.ReLU()
        )
        
    def forward(self, x):
        # 计算FFT
        fft = torch.fft.rfft(x, dim=2)
        fft_magnitude = torch.abs(fft)  # 获取幅度
        
        # 取指定数量的频率点
        if fft_magnitude.shape[2] > self.freq_bins:
            fft_features = fft_magnitude[:, :, :self.freq_bins]
        else:
            # 如果频率点不足，进行填充
            padding_size = self.freq_bins - fft_magnitude.shape[2]
            fft_features = F.pad(fft_magnitude, (0, padding_size))
        
        # 压缩通道维度
        fft_features = torch.mean(fft_features, dim=1)  # [batch_size, freq_bins]
        
        # 投影到所需维度
        fft_features = self.fft_projection(fft_features)  # [batch_size, d_model]
        
        return fft_features

class InceptionModule(nn.Module):
    """
    Inception模块，实现多尺度特征提取
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        branch_channels = out_channels // 4
        
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm1d(branch_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        branch3_output = self.branch3(x)
        branch4_output = self.branch4(x)
        
        outputs = [branch1_output, branch2_output, branch3_output, branch4_output]
        return torch.cat(outputs, 1)

class VibrationFeatureExtractor(nn.Module):
    """
    振动信号特征提取器，集成多种先进特征提取方法
    
    参数:
        in_channels (int): 输入通道数，默认为1
        base_filters (int): 基础滤波器数量，默认为32
        d_model (int): 模型维度，默认为768
        seq_len (int): 输出序列长度，默认为16
        use_fft (bool): 是否使用FFT特征，默认为True
    """
    def __init__(self, in_channels=1, base_filters=32, d_model=768, seq_len=16, use_fft=True):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.use_fft = use_fft
        
        # 初始卷积层
        self.initial_conv = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Inception模块
        self.inception = InceptionModule(base_filters, base_filters * 4)
        
        # 残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(base_filters * 4, base_filters * 8),
            ResidualBlock(base_filters * 8, base_filters * 16)
        ])
        
        # SE注意力模块
        self.se_module = SEModule(base_filters * 16)
        
        # 时间维度压缩
        self.pool = nn.AdaptiveAvgPool1d(seq_len)
        
        # 特征维度投影
        self.projection = nn.Conv1d(base_filters * 16, d_model, kernel_size=1)
        
        # 频域特征提取（可选）
        if use_fft:
            self.fft_layer = FFTFeatureLayer(d_model)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征，形状为[batch_size, channels, seq_len]
            
        返回:
            特征序列，形状为[batch_size, seq_len, d_model]
        """
        # 确保输入形状正确 [batch_size, channels, time_steps]
        if x.dim() == 3 and x.size(1) != 1:
            # 如果第二个维度不是channels，而是时间步，则转置
            if x.size(2) == 1:
                x = x.transpose(1, 2)
        
        # 初始特征提取
        x = self.initial_conv(x)
        
        # Inception特征提取
        x = self.inception(x)
        
        # 残差块特征提取
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # 注意力增强
        x = self.se_module(x)
        
        # 时间维度调整
        x = self.pool(x)  # [batch_size, channels, seq_len]
        
        # 特征维度投影
        x = self.projection(x)  # [batch_size, d_model, seq_len]
        
        # 融合频域特征（如果启用）
        if self.use_fft:
            fft_features = self.fft_layer(x)  # [batch_size, d_model]
            # 扩展维度以匹配时间序列
            fft_features = fft_features.unsqueeze(1).expand(-1, self.seq_len, -1)  # [batch_size, seq_len, d_model]
        
        # 转换维度顺序以适配Transformer [batch_size, seq_len, d_model]
        x = x.transpose(1, 2)
        
        # 如果使用FFT特征，则进行融合
        if self.use_fft:
            x = x + 0.1 * fft_features  # 加权融合
        
        return x 