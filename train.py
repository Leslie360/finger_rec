import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import os
import logging
import sys
import matplotlib
matplotlib.use('Agg')  # 强制使用非交互式后端，防止在无界面的 Docker/WSL 中报错
import matplotlib.pyplot as plt

from config import (
    BATCH_SIZE, AMP_ENABLED, LTP_POLY, LTD_POLY,
    LEARNING_RATE, NUM_EPOCHS, WARMUP_EPOCHS,
    LOG_DIR, LOG_FILE, MODEL_SAVE_PATH
)
from dataset import get_dataloaders
from model import get_organic_resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logging():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    log_path = os.path.join(LOG_DIR, LOG_FILE)
    
    # 重新配置 Logger，避免重复打印
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 清空之前的 handler
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # 文件 Handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台 Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def plot_curves(train_losses, test_losses, train_accs, test_accs):
    """绘制 Loss 和 Accuracy 曲线并保存"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss 子图
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy 子图
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Acc')
    plt.plot(epochs, test_accs, 'r-', label='Test Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(LOG_DIR, "loss_acc_curves.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Training curves saved to {save_path}")

def apply_ltp_ltd_nonlinearity(model):
    """硬件感知梯度修正"""
    for param in model.parameters():
        if param.grad is not None and param.requires_grad:
            if param.ndim < 2: continue
            
            g_old = param.grad
            w = param.data
            
            # Normalize w to [0, 1]
            w_norm = (w + 1) / 2
            w_norm = torch.clamp(w_norm, 0, 1)
            
            # Polyval helper
            def torch_polyval(coeffs, x):
                return (coeffs[0] * x**3 + coeffs[1] * x**2 + 
                        coeffs[2] * x + coeffs[3])
            
            ltp_coeffs = torch.tensor(LTP_POLY, device=param.device, dtype=param.dtype)
            ltd_coeffs = torch.tensor(LTD_POLY, device=param.device, dtype=param.dtype)
            
            slope_ltp = torch_polyval(ltp_coeffs, w_norm)
            slope_ltd = torch_polyval(ltd_coeffs, w_norm)
            
            slope = torch.where(g_old < 0, slope_ltp, slope_ltd)
            
            # Update: g_new = g_old * 0.7 + (g_old * slope) * 0.3
            g_new = g_old * 0.7 + (g_old * slope) * 0.3
            param.grad = g_new

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            progress = min(max(progress, 0.0), 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [base_lr * cosine_decay for base_lr in self.base_lrs]

def train_one_epoch(model, loader, optimizer, criterion, scaler):
    """训练一个 Epoch，不打印每一步，只返回统计信息"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=AMP_ENABLED, dtype=torch.bfloat16):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        apply_ltp_ltd_nonlinearity(model)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Hard Clamp
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.data.clamp_(-1, 1)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(loader)
    avg_acc = 100. * correct / total
    return avg_loss, avg_acc

def evaluate(model, loader, criterion):
    """评估模型，返回 Loss 和 Acc"""
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
    
    avg_loss = running_loss / len(loader)
    avg_acc = 100. * correct / total
    return avg_loss, avg_acc

def main():
    setup_logging()
    logging.info(f"Configuration: LR={LEARNING_RATE}, Epochs={NUM_EPOCHS}, Batch={BATCH_SIZE}, AMP={AMP_ENABLED}")
    
    train_loader, test_loader = get_dataloaders()
    
    model = get_organic_resnet18(pretrained=True)
    model = model.to(device)
    
    try:
        model = torch.compile(model)
        logging.info("Model compiled successfully.")
    except Exception as e:
        logging.warning(f"Compile failed: {e}")
        
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=AMP_ENABLED)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)
    
    # 存储曲线数据
    history = {
        'train_loss': [], 'test_loss': [],
        'train_acc': [], 'test_acc': []
    }
    
    logging.info("Start Training...")
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # 1. 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        
        # 2. 测试
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        
        # 3. 调度器步进
        scheduler.step()
        
        duration = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # 4. 记录数据
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        # 5. 打印一行清晰的日志
        logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Time: {duration:.1f}s | "
                     f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                     f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% | "
                     f"LR: {current_lr:.6f}")
        
    # 保存模型
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logging.info(f"Training complete. Model saved to {MODEL_SAVE_PATH}")
    
    # 绘制曲线
    plot_curves(history['train_loss'], history['test_loss'], 
                history['train_acc'], history['test_acc'])

if __name__ == "__main__":
    main()