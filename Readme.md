# FPGA Quantized Neural Network Accelerator (LeNet-MNIST)

本项目利用 Brevitas 进行量化感知训练 (QAT)，并通过 Xilinx FINN 编译器将模型部署到 Zynq FPGA 上。项目实现了从模型训练、硬件编译到上板验证的完整流程。

## 项目结构说明 (Directory Structure)

根据当前目录布局，各文件夹功能描述如下：

### training/: 

核心训练源码。包含基于 Brevitas 定义的量化模型（如 Letnet.py）、数据预处理逻辑以及 PyTorch 训练脚本。

### notebooks/: 

FINN 编译与验证流程。存放 Jupyter Notebook 文件，记录了从 ONNX 导出、算子变换、RTL 仿真验证到最后生成硬件比特流的每一个步骤。

### models/: 

模型权重与中间产物，存放训练生成的 .onnx 量化图文件以及FINN中间处理的.onnx量化图文件。

### deployment/: 硬件部署资源。

- top_wrapper.bit: 生成的 FPGA 比特流文件。

- top.hwh: 硬件描述文件（PYNQ 运行环境必需）。

- driver.py: 在 PYNQ 开发板上调起硬件加速器并进行推理测试的驱动脚本。

### Doc/: 

项目文档，存放参考文件，实验报告，分工表等。

### figure/: 

运行结果截图，存放训练准确率曲线、Loss 曲线、Netron 模型图以及 README 中使用的各类截图。
