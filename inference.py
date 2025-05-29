import torch
import torchvision
import torchvision.transforms as transforms
import time
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from modules.resnet_18_baseline_fp32 import ResNet18 as ResNet18_baseline

# --------------------------
# 参数配置
# --------------------------
MODEL_PATH = "./pytorch/model/net_123.pth"
DATA_ROOT = "./pytorch/data"
MODEL_DTYPE = "FP32" 
BATCH_SIZES = [8, 16, 32, 64, 128, 256, 512]  
NUM_WORKERS = 4
RUN_TIMES = 1 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if MODEL_DTYPE == "FP16":
    from modules.resnet_18_optim_fp16 import ResNet18 as ResNet18_optim
elif MODEL_DTYPE == "FP32":
    from modules.resnet_18_optim_fp32 import ResNet18 as ResNet18_optim
else:
    from modules.resnet_18_baseline_fp32 import ResNet18 as ResNet18_optim

# --------------------------
# 模型加载
# --------------------------
def load_model(model_path, model_type='baseline', model_dtype='FP32'):
    if model_type == 'optim':
        model = ResNet18_optim()
    elif model_type == 'baseline':
        model = ResNet18_baseline()
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint)
    
    dtype = torch.half if model_dtype == "FP16" else torch.float32
    model = model.to(DEVICE, dtype=dtype)
    model.eval()
    return model

# --------------------------
# 推理性能测试函数
# --------------------------
def process_single_run(model, dataloader, device, model_dtype='FP32'):
    warmup_dtype = torch.half if model_dtype == "FP16" else torch.float32
    total_images = len(dataloader.dataset)
    
    with torch.no_grad():
        warmup_tensor = torch.randn(dataloader.batch_size, 3, 32, 32, 
                                   dtype=warmup_dtype).to(device)
        model(warmup_tensor)  
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    with torch.no_grad():
        progress_bar = tqdm(
            dataloader,
            desc=f"reasoning (bs={dataloader.batch_size})",
            ncols=100, 
            bar_format="{l_bar}{bar} [{elapsed}<{remaining}]"
        )
        
        for images, _ in progress_bar:
            images = images.to(device=device, dtype=warmup_dtype)
            _ = model(images)
            
            processed = (progress_bar.n + 1) * dataloader.batch_size
            current_speed = processed / (time.perf_counter() - start_time)
            progress_bar.set_postfix({
                "speed": f"{current_speed:.1f} img/s"
            })

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    
    return total_images / elapsed


if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    testset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform
    )
    model_optim = load_model(MODEL_PATH, 'optim', MODEL_DTYPE)
    model_baseline = load_model(MODEL_PATH, 'baseline', 'FP32')

    subset_indices = range(50)
    subset = torch.utils.data.Subset(testset, subset_indices)
    subset_loader = DataLoader(subset, batch_size=50, shuffle=False)
    def denormalize(tensor):
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.247, 0.243, 0.261]).view(3, 1, 1)
        return tensor.cpu() * std + mean

    with torch.no_grad():
        images, labels = next(iter(subset_loader))
        images = images.to(DEVICE, dtype=torch.half if MODEL_DTYPE == "FP16" else torch.float32)
        outputs = model_optim(images)
        preds = outputs.argmax(dim=1).cpu()

    correct = (preds == labels).sum().item()
    total = len(labels)
    accuracy = correct / total * 100
    print(f"\nAccuracy Breakdown:")
    print(f"-------------------")
    print(f"Correct predictions: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2f}%\n")

    plt.figure(figsize=(15, 12))
    for i in range(50):
        plt.subplot(5, 10, i+1)
        
        img = denormalize(images[i].cpu().float()).clamp(0, 1)
        plt.imshow(img.permute(1, 2, 0))  
        
        true_label = testset.classes[labels[i]]
        pred_label = testset.classes[preds[i]]
        is_correct = true_label == pred_label
        
        status = "✓" if is_correct else "✗"
        print(f"Sample {i+1:2d}: Pred={pred_label:9s} | True={true_label:9s} | {status}")
        
        title_color = "red" if not is_correct else "black"
        plt.title(f"Label: {true_label}\nPred: {pred_label}", 
                 color=title_color, fontsize=9)
        plt.axis('off')
    
    plt.suptitle("Optimized Model Prediction Visualization (First 50 Samples)", 
                y=0.99, fontsize=14)
    plt.tight_layout()
    plt.show()

    optim_throughputs = []
    baseline_throughput = None

    baseline_loader = DataLoader(
        testset,
        batch_size=128,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    print(f"Testing baseline:")
    baseline_throughput = process_single_run(
        model=model_baseline,
        dataloader=baseline_loader,
        device=DEVICE,
        model_dtype='FP32'
    )
    print(f"[Baseline] Batch Size=128 | Throughput: {baseline_throughput:.2f} img/s")

    for batch_size in BATCH_SIZES:
        
        testloader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        
        throughputs = []
        print(f"\nTesting batch_size = {batch_size}")
        for run in range(RUN_TIMES):
            throughput = process_single_run(
                model=model_optim,
                dataloader=testloader,
                device=DEVICE,
                model_dtype=MODEL_DTYPE
            )
            throughputs.append(throughput)
        
        avg_throughput = sum(throughputs) / RUN_TIMES
        optim_throughputs.append(avg_throughput)
        print(f"[Optim] Batch Size={batch_size:3d} | Average Throughput: {avg_throughput:.2f} img/s")

    plt.figure(figsize=(12, 6))
    
    plt.plot(BATCH_SIZES, optim_throughputs, 
            marker='o', linestyle='-', color='#FF6F00', 
            linewidth=2, markersize=10, label='Optimized Model')
    
    plt.axhline(y=baseline_throughput, color='#1F77B4', linestyle='--', 
                linewidth=2, label=f'Baseline (Batch Size=128)')
    
    plt.scatter([128], [optim_throughputs[-1]], color='red', zorder=5, 
                label=f'Optimized @128: {optim_throughputs[-1]:.1f} img/s')
    
    plt.title('Optimized Model Throughput vs Baseline (Batch Size=128)', fontsize=14, pad=20)
    plt.xlabel('Batch Size', fontsize=12, labelpad=10)
    plt.ylabel('Throughput (images/sec)', fontsize=12, labelpad=10)
    plt.xticks(BATCH_SIZES, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='upper left')
    
    for x, y in zip(BATCH_SIZES, optim_throughputs):
        plt.text(x, y+50, f'{y:.1f}', ha='center', va='bottom', fontsize=10, color='#FF6F00')
    
    plt.tight_layout()
    plt.show()
