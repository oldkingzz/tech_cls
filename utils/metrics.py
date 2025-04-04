import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def calculate_metrics(y_pred, y_true):
    """
    计算RUL预测的评估指标
    
    参数:
        y_pred (np.ndarray): 预测值
        y_true (np.ndarray): 真实值
        
    返回:
        dict: 包含各项评估指标的字典
    """
    # 确保输入是numpy数组
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # 计算基本指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # 计算PHM2008评分
    diff = y_pred - y_true
    score = np.zeros_like(diff)
    
    # 过早预测惩罚（预测值小于真实值）
    early_mask = diff < 0
    score[early_mask] = np.exp(-diff[early_mask] / 10) - 1
    
    # 过晚预测惩罚（预测值大于真实值）
    late_mask = diff >= 0
    score[late_mask] = np.exp(diff[late_mask] / 13) - 1
    
    phm_score = np.mean(score)
    
    # 计算相对误差
    relative_error = np.abs(diff) / (y_true + 1e-6)  # 添加小值避免除零
    mre = np.mean(relative_error)  # 平均相对误差
    
    # 计算准确度指标
    accuracy_10 = np.mean(np.abs(diff) <= 10)  # 预测误差在±10范围内的比例
    accuracy_20 = np.mean(np.abs(diff) <= 20)  # 预测误差在±20范围内的比例
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'phm_score': float(phm_score),
        'mre': float(mre),
        'accuracy_10': float(accuracy_10),
        'accuracy_20': float(accuracy_20)
    }
