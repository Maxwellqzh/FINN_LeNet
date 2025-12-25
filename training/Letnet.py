import os
import sys

# 1. 解决 OMP 冲突 (必须在最前面)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import brevitas.nn as qnn
import numpy as np
from brevitas.inject.enum import ScalingImplType
from brevitas.export import export_qonnx
from brevitas.core.zero_point import ZeroZeroPoint 

# [关键修复] 引入所有底层依赖，不再依赖自动推断
from brevitas.inject import ExtendedInjector
from brevitas.core.quant import BinaryQuant  # 核心量化算法
from brevitas.core.scaling import ConstScaling # 缩放实现
from brevitas.core.restrict_val import RestrictValueType # 约束类型
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector # 代理类
from types import ModuleType

# --- 引入 ONNX 推理库 ---
try:
    import onnxruntime as ort
    ONNX_RUNTIME_AVAILABLE = True
except ImportError:
    print("⚠️ 警告: 未安装 onnxruntime. 无法进行 ONNX 模型推理验证.")
    ONNX_RUNTIME_AVAILABLE = False

# --- 引入 matplotlib 用于绘图 ---
try:
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("⚠️ 警告: 未安装 matplotlib. 无法绘制训练曲线.")
    MATPLOTLIB_AVAILABLE = False
# -------------------------

# 2. 解决 ONNXOptimizer 缺失 (FINN环境兼容性修复)
fake_opt = ModuleType("onnxoptimizer")
fake_opt.optimize = lambda model, passes=None, fixed_point=False: model
sys.modules["onnxoptimizer"] = fake_opt

# 3. 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 使用设备: {device} | 启动 W2A1 对称量化训练...")

# ==============================================================================
# [终极修复] 手动全配置 Bipolar 量化器
# 显式定义所有属性，防止 NoneType 错误
# ==============================================================================
class CommonBinaryActQuant(ExtendedInjector):
    # 1. 结构依赖
    proxy_class = ActQuantProxyFromInjector
    tensor_quant = BinaryQuant
    scaling_impl = ConstScaling
    
    # 2. 参数配置
    bit_width = 1
    min_val = -1.0
    max_val = 1.0
    scaling_init = 1.0
    restrict_scaling_type = RestrictValueType.FP
    
    # [关键修复] 显式声明符号属性，解决 TypeError: 'NoneType'
    signed = True
    is_signed = True
    
    # 其他
    return_quant_tensor = False
    zero_point_impl = ZeroZeroPoint

# ==============================================================================
# 2. 网络定义：核心 W2A1 (权重2bit, 激活1bit Bipolar)
# ==============================================================================
class LeNet_W2A1_MixedPrecision(nn.Module):
    def __init__(self):
        super(LeNet_W2A1_MixedPrecision, self).__init__()
        
        # =======================================================
        # 1. 激活配置 (Act)
        # =======================================================
        kwargs_act_input = {
            'quant_type': 'INT', 'bit_width': 2, 
            'min_val': 0.0, 'max_val': 1.0, 
            'scaling_impl_type': ScalingImplType.CONST, 'scaling_const': 1.0,
            'return_quant_tensor': False, 'zero_point_impl': ZeroZeroPoint 
        }
        
        # 核心激活
        kwargs_act_core = {
            'act_quant': CommonBinaryActQuant,
            'return_quant_tensor': False 
        }

        # =======================================================
        # 2. 权重配置 (Weight)
        # =======================================================
        base_weight_kwargs = {
            'quant_type': 'INT', 
            'scaling_impl_type': ScalingImplType.CONST, 
            'scaling_const': 1.0,
            'return_quant_tensor': False, 
            'bias': False, 
            'narrow_range': True, 
            'zero_point_impl': ZeroZeroPoint
        }

        kwargs_weight_core = base_weight_kwargs.copy()
        kwargs_weight_core['weight_bit_width'] = 2 

        kwargs_weight_begin = base_weight_kwargs.copy()
        kwargs_weight_begin['weight_bit_width'] = 2

        kwargs_weight_end = base_weight_kwargs.copy()
        kwargs_weight_end['weight_bit_width'] = 2

        # =======================================================
        # 3. 网络层定义
        # =======================================================
        self.quant_input = qnn.QuantIdentity(**kwargs_act_input)

        # Layer 1: Conv1
        self.conv1 = qnn.QuantConv2d(1, 8, kernel_size=5, padding=0, **kwargs_weight_begin)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-3)
        self.act1 = qnn.QuantIdentity(**kwargs_act_core) 
        self.pool1 = nn.MaxPool2d(2, 2)

        # Layer 2: Conv2
        self.conv2 = qnn.QuantConv2d(8, 16, kernel_size=5, padding=0, **kwargs_weight_core)
        self.bn2 = nn.BatchNorm2d(16, eps=1e-3)
        self.act2 = qnn.QuantIdentity(**kwargs_act_core)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Layer 3: Conv3
        self.conv3 = qnn.QuantConv2d(16, 32, kernel_size=5, padding=0, **kwargs_weight_core)
        self.bn3 = nn.BatchNorm2d(32, eps=1e-3)
        self.act3 = qnn.QuantIdentity(**kwargs_act_core)

        # FC 1: FC1
        self.fc1 = qnn.QuantLinear(32, 16, **kwargs_weight_core)
        self.act4 = qnn.QuantIdentity(**kwargs_act_core)

        # FC 2: FC2 输出层
        self.fc2 = qnn.QuantLinear(16, 10, output_quant=None, **kwargs_weight_end)

        self._init_weights()

    def _init_weights(self):
        print("🔧 初始化权重 (Uniform -0.8 ~ 0.8)...")
        for m in self.modules():
            if isinstance(m, (qnn.QuantConv2d, qnn.QuantLinear)):
                nn.init.uniform_(m.weight, -0.8, 0.8)

    def forward(self, x):
        x = self.quant_input(x)
        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.pool2(self.act2(self.bn2(self.conv2(x))))
        x = self.act3(self.bn3(self.conv3(x)))
        x = x.view(x.shape[0], -1)
        x = self.act4(self.fc1(x))
        x = self.fc2(x)
        return x

# ==============================================================================
# 3. 评估函数
# ==============================================================================
def evaluate_model(model, data_loader, criterion, mode="验证"):
    model.eval()
    total_loss = 0
    correct = 0
    dataset_size = len(data_loader.dataset)
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item() 
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_loss = total_loss / dataset_size
    accuracy = 100. * correct / dataset_size

    print(f"\n🔬 PyTorch {mode}集结果: 平均损失={avg_loss:.4f}, 准确率={correct}/{dataset_size} ({accuracy:.2f}%)")
    return avg_loss, accuracy

# ==============================================================================
# 4. ONNX 推理测试函数
# ==============================================================================
def test_onnx_model(onnx_path, test_loader):
    if not ONNX_RUNTIME_AVAILABLE:
        print("❌ 无法进行 ONNX 推理验证，请安装 onnxruntime。")
        return
    print(f"\n🔎 启动 ONNX 模型 ({onnx_path}) 精度验证...")

    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    
    correct = 0
    total = 0
    for data, target in test_loader:
        data_np = data.cpu().numpy().astype(np.float32)
        ort_inputs = {input_name: data_np}
        ort_outputs = ort_session.run([output_name], ort_inputs)
        output = torch.from_numpy(np.array(ort_outputs[0]))
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)

    accuracy = 100. * correct / total
    print(f"✅ ONNX Runtime 测试结果: 准确率={correct}/{total} ({accuracy:.2f}%)")

# ==============================================================================
# 5. 主训练流程
# ==============================================================================
def train_symmetric_final():

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_val_dataset_full = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    val_split_ratio = 0.1
    val_size = int(val_split_ratio * len(train_val_dataset_full))
    train_size = len(train_val_dataset_full) - val_size
    
    print(f"🔄 划分数据集: 训练集 {train_size} 张, 验证集 {val_size} 张.")
    train_dataset, val_dataset = random_split(train_val_dataset_full, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    model = LeNet_W2A1_MixedPrecision().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005) 
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    num_epochs = 5
    best_val_acc = 0.0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        print(f"\n=================== Epoch {epoch+1}/{num_epochs} Start ===================")

        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            epoch_train_correct += pred.eq(target.view_as(pred)).sum().item()
            epoch_train_total += batch_size

            if batch_idx % 100 == 0:
                print(f"[Step {batch_idx}] Loss={loss.item()/batch_size:.4f}")

        avg_train_loss = epoch_train_loss / epoch_train_total
        train_accuracy = 100. * epoch_train_correct / epoch_train_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)

        val_loss, val_acc = evaluate_model(model, val_loader, criterion, mode="验证")
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"🏆 新的最佳验证准确率: {best_val_acc:.2f}%，模型已更新。")

        print(f"=================== Epoch {epoch+1} Complete (Val Acc: {val_acc:.2f}%) ===================")

    print(f"\n{'#'*60}")
    print("🎯 训练完成！开始最终测试 (使用独立的测试集)...")
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, mode="最终测试")
    print(f"################### 最终测试准确率 (PyTorch): {test_acc:.2f}% ###################")

    if MATPLOTLIB_AVAILABLE:
        print("\n📊 绘制训练过程曲线...")
        epochs = range(1, num_epochs + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Loss')
        ax1.legend()
        ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Accuracy')
        ax2.legend()
        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.show()

    output_path = "lenet_w2a1_mixed.onnx"
    print(f"\n📤 导出ONNX模型: {output_path}")
    model.eval()
    model_cpu = model.to('cpu')
    dummy_input = torch.rand(1, 1, 32, 32).to('cpu') 
    
    try:
        export_qonnx(model_cpu, dummy_input, output_path, opset_version=11)
        print(f"🎉 成功导出: {output_path}")
        print("💡 提示: 模型现已包含 Bipolar 属性 (bias=-1, scale=2)。")
    except Exception as e:
        print(f"❌ 导出ONNX失败: {e}")

    if ONNX_RUNTIME_AVAILABLE:
        test_onnx_model(output_path, test_loader)
    
if __name__ == '__main__':
    train_symmetric_final()