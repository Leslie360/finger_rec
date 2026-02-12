import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import os
import logging
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import (
    BATCH_SIZE, AMP_ENABLED, LTP_POLY, LTD_POLY,
    DEVICE_NOISE_STD, NORM_MEAN, NORM_STD
)
from dataset import get_dataloaders
from model import get_organic_resnet18

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 训练逻辑 (保持不变)
# ==========================================

def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def apply_ltp_ltd_nonlinearity(model):
    for param in model.parameters():
        if param.grad is not None and param.requires_grad and param.ndim >= 2:
            g_old = param.grad
            w = param.data
            w_norm = torch.clamp((w + 1) / 2, 0, 1)
            
            def torch_polyval(coeffs, x):
                return coeffs[0] * x**3 + coeffs[1] * x**2 + coeffs[2] * x + coeffs[3]
            
            ltp_coeffs = torch.tensor(LTP_POLY, device=param.device, dtype=param.dtype)
            ltd_coeffs = torch.tensor(LTD_POLY, device=param.device, dtype=param.dtype)
            
            slope = torch.where(g_old < 0, torch_polyval(ltp_coeffs, w_norm), torch_polyval(ltd_coeffs, w_norm))
            param.grad = g_old * 0.7 + (g_old * slope) * 0.3

def train_one_epoch(model, loader, optimizer, criterion, scaler, epoch, args, is_calibration=False):
    model.train()
    
    if is_calibration and args.enable_calibration:
        for module in model.modules():
            if hasattr(module, 'noise_std'):
                module.noise_std = 0.0
    
    if args.enable_dynamic_noise and not is_calibration:
        if epoch < args.epochs * 0.8:
            noise_mult = np.random.uniform(0.5, 2.5)
        else:
            noise_mult = np.random.uniform(0.5, 1.0)
        
        current_noise = DEVICE_NOISE_STD * noise_mult
        for module in model.modules():
            if hasattr(module, 'noise_std'):
                module.noise_std = current_noise
    elif not is_calibration:
        for module in model.modules():
            if hasattr(module, 'noise_std'):
                module.noise_std = DEVICE_NOISE_STD

    running_loss = 0.0
    correct = 0
    total = 0
    
    use_mixup = args.enable_mixup and (epoch >= args.warmup_epochs) and (not is_calibration)
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
            
        with torch.amp.autocast('cuda', enabled=AMP_ENABLED, dtype=torch.bfloat16):
            outputs = model(inputs)
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        if not is_calibration:
            apply_ltp_ltd_nonlinearity(model)
            
        scaler.step(optimizer)
        scaler.update()
        
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.data.clamp_(-1, 1)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / len(loader), 100. * correct / total

# ==========================================
# 数据导出与可视化 (Data Export & Viz)
# ==========================================

def save_txt_data(filepath, header, data):
    """通用保存 txt 函数"""
    try:
        np.savetxt(filepath, data, header=header, delimiter=',', comments='', fmt='%.6f')
        logging.info(f"Data saved to {filepath}")
    except Exception as e:
        logging.error(f"Failed to save data to {filepath}: {e}")

def plot_and_save_curves(history, log_dir):
    """绘制并保存训练曲线数据"""
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    # 1. 绘图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train')
    plt.plot(epochs_range, history['test_loss'], label='Test')
    plt.title('Loss'); plt.legend(); plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train')
    plt.plot(epochs_range, history['test_acc'], label='Test')
    plt.title('Acc'); plt.legend(); plt.grid(True)
    
    plt.savefig(os.path.join(log_dir, "curves.png"))
    plt.close()
    
    # 2. 保存数据到 txt
    # 格式: epoch, train_loss, test_loss, train_acc, test_acc
    data = np.stack([
        epochs_range, 
        history['train_loss'], 
        history['test_loss'], 
        history['train_acc'], 
        history['test_acc']
    ], axis=1)
    
    save_txt_data(
        os.path.join(log_dir, "history_curves.txt"), 
        "epoch,train_loss,test_loss,train_acc,test_acc", 
        data
    )

def plot_and_save_weights(init_weights, final_model, log_dir):
    """绘制并保存权重数据"""
    final_weights = []
    for name, param in final_model.named_parameters():
        if "conv1.weight" in name:
            final_weights = param.data.cpu().numpy()
            break
    if len(final_weights) == 0: return

    flat_init = init_weights.flatten()
    flat_final = final_weights.flatten()
    
    # 1. 绘图 (直方图 + 热力图)
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    sns.histplot(flat_init, bins=50, color='gray', kde=True, label='Init', stat='density', alpha=0.5)
    plt.title("Weight Distribution: Init"); plt.legend()
    
    plt.subplot(2, 2, 2)
    sns.histplot(flat_final, bins=50, color='purple', kde=True, label='Trained', stat='density', alpha=0.5)
    plt.title("Weight Distribution: Trained"); plt.legend()
    
    k_init = np.mean(init_weights[:4], axis=1) 
    k_final = np.mean(final_weights[:4], axis=1)
    viz_init = np.hstack([k_init[i] for i in range(4)])
    viz_final = np.hstack([k_final[i] for i in range(4)])
    
    plt.subplot(2, 1, 2)
    plt.imshow(np.vstack([viz_init, viz_final]), cmap='coolwarm')
    plt.title("Kernel Heatmap (Top: Init, Bottom: Trained)"); plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "weight_contrast.png"))
    plt.close()
    
    # 2. 保存数据到 txt
    # 为了方便画分布图，直接保存展平后的权重
    # 注意：数据量可能较大，这里只保存 conv1 的所有权重
    np.savetxt(os.path.join(log_dir, "weights_init_conv1.txt"), flat_init, fmt='%.6f')
    np.savetxt(os.path.join(log_dir, "weights_trained_conv1.txt"), flat_final, fmt='%.6f')
    logging.info(f"Weight data saved to weights_init_conv1.txt and weights_trained_conv1.txt")

def save_confusion_matrix(model, loader, log_dir):
    """生成并保存混淆矩阵数据"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    cm = confusion_matrix(all_labels, all_preds)
    
    # 保存图片
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(log_dir, "confusion_matrix.png"))
    plt.close()
    
    # 保存数据 txt
    np.savetxt(os.path.join(log_dir, "confusion_matrix.txt"), cm, fmt='%d')
    logging.info("Confusion matrix data saved.")

def robust_evaluate(model, loader, noise_mult=1.0, samples=5):
    """辅助函数：执行鲁棒性测试并返回准确率"""
    model.eval()
    current_noise = DEVICE_NOISE_STD * noise_mult
    for module in model.modules():
        if hasattr(module, 'noise_std'):
            module.noise_std = current_noise
            module.train() 
            
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs_sum = 0
            for _ in range(samples):
                outputs_sum += model(inputs)
            outputs_avg = outputs_sum / samples
            _, predicted = outputs_avg.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

def run_final_robustness_scan(model, loader, log_dir):
    """运行最终的鲁棒性扫描并保存数据"""
    logging.info("\n=== Final Robustness Evaluation ===")
    
    # 1. 扫描不同噪声水平 (单次采样)
    noise_levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    scan_results = []
    
    logging.info("Scanning noise levels (Single Shot)...")
    for nl in noise_levels:
        acc = robust_evaluate(model, loader, noise_mult=nl, samples=1)
        scan_results.append([nl, acc])
        logging.info(f"Noise x{nl}: {acc:.2f}%")
        
    # 保存扫描数据
    scan_data = np.array(scan_results)
    save_txt_data(
        os.path.join(log_dir, "robustness_scan_data.txt"),
        "noise_multiplier,accuracy_percent",
        scan_data
    )
    
    # 2. 关键点多次采样测试
    logging.info("Multi-sampling tests...")
    ms_results = []
    
    # Noise x1.0, Samples 1
    acc_1_1 = robust_evaluate(model, loader, noise_mult=1.0, samples=1)
    ms_results.append([1.0, 1, acc_1_1])
    logging.info(f"Noise x1.0 (1 sample): {acc_1_1:.2f}%")
    
    # Noise x1.0, Samples 10
    acc_1_10 = robust_evaluate(model, loader, noise_mult=1.0, samples=10)
    ms_results.append([1.0, 10, acc_1_10])
    logging.info(f"Noise x1.0 (10 samples): {acc_1_10:.2f}%")
    
    # Noise x1.5, Samples 10
    acc_15_10 = robust_evaluate(model, loader, noise_mult=1.5, samples=10)
    ms_results.append([1.5, 10, acc_15_10])
    logging.info(f"Noise x1.5 (10 samples): {acc_15_10:.2f}%")
    
    # 保存多次采样数据
    save_txt_data(
        os.path.join(log_dir, "multisample_data.txt"),
        "noise_multiplier,samples,accuracy_percent",
        np.array(ms_results)
    )

# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='baseline', help='Experiment name')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--enable_mixup', action='store_true')
    parser.add_argument('--enable_calibration', action='store_true')
    parser.add_argument('--enable_dynamic_noise', action='store_true')
    args = parser.parse_args()

    log_dir = os.path.join("logs", args.exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"=== Experiment: {args.exp_name} ===")
    logging.info(f"Settings: Mixup={args.enable_mixup}, Calib={args.enable_calibration}, DynNoise={args.enable_dynamic_noise}")
    
    set_seed(42)
    train_loader, test_loader = get_dataloaders()
    
    model = get_organic_resnet18(pretrained=True).to(device)
    init_weights = model.conv1.weight.data.cpu().numpy().copy()
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=AMP_ENABLED)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}

    try:
        model = torch.compile(model)
        logging.info("Model compiled.")
    except:
        pass

    for epoch in range(args.epochs):
        start = time.time()
        
        cycle_idx = epoch % 20
        is_calib = (cycle_idx >= 18) if args.enable_calibration else False
        
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, epoch, args, is_calib)
        v_loss, v_acc = evaluate(model, test_loader, criterion)
        scheduler.step()
        
        history['train_loss'].append(t_loss)
        history['test_loss'].append(v_loss)
        history['train_acc'].append(t_acc)
        history['test_acc'].append(v_acc)
        
        calib_str = "[CALIB]" if is_calib else ""
        logging.info(f"Ep {epoch+1}/{args.epochs} {calib_str} | T_Loss:{t_loss:.4f} Acc:{t_acc:.2f}% | V_Loss:{v_loss:.4f} Acc:{v_acc:.2f}% | Time:{time.time()-start:.1f}s")

    torch.save(model.state_dict(), os.path.join(log_dir, "model.pth"))
    
    # === 保存所有数据和图表 ===
    logging.info("Saving plots and raw data...")
    plot_and_save_curves(history, log_dir)
    plot_and_save_weights(init_weights, model, log_dir)
    save_confusion_matrix(model, test_loader, log_dir)
    run_final_robustness_scan(model, test_loader, log_dir)
    
    logging.info("Experiment Finished.")

if __name__ == "__main__":
    main()