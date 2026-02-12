import subprocess
import os
import time

def run_cmd(cmd):
    print(f"\n[Running Command]: {cmd}")
    # 使用 subprocess.call 确保前一个跑完才跑下一个
    subprocess.check_call(cmd, shell=True)

def main():
    print("=== Starting Ablation Study for Fingerprint Recognition ===\n")
    
    # Experiment 1: Baseline (无Mixup, 无Cycle, 无动态噪声)
    # 预期: Ideal Acc 高, Robustness 极差
    print(">>> Experiment 1/3: Baseline (Ideal but Fragile)")
    run_cmd("python3 train.py --exp_name exp1_baseline --epochs 200")
    
    # Experiment 2: Method 1 (开启Mixup 和 Calibration, 但无动态噪声)
    # 预期: 收敛更好, 过拟合消失, 但抗大噪声能力一般
    print("\n>>> Experiment 2/3: Algorithmic (Mixup + Calibration)")
    run_cmd("python3 train.py --exp_name exp2_algo --epochs 200 --enable_mixup --enable_calibration")
    
    # Experiment 3: Method 2 (开启所有功能 + 动态噪声)
    # 预期: Ideal Acc 略降 (92-93%), Robustness 极强
    print("\n>>> Experiment 3/3: Full Proposed (Hardware Aware)")
    run_cmd("python3 train.py --exp_name exp3_full --epochs 200 --enable_mixup --enable_calibration --enable_dynamic_noise")
    
    print("\n=== All Experiments Completed ===")
    print("Check 'logs/' directory for results.")

if __name__ == "__main__":
    main()