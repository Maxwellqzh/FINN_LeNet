# FPGA量化神经网络加速器 (LeNet-MNIST)

本项目利用 Brevitas 进行量化感知训练 (QAT)，并通过 Xilinx FINN 编译器将模型部署到 Zynq FPGA 上。项目实现了从模型训练、硬件编译到上板验证的完整流程。

## 项目结构说明 (Directory Structure)

根据当前的文件列表，各文件夹的具体组成和功能如下：

### 1. deployment (硬件部署资源)

存放用于 PYNQ 开发板部署的最终产物。

* top_wrapper.bit: 生成的 FPGA 比特流文件。
* top.hwh: 硬件描述文件，用于 PYNQ 建立 Overlay。
* driver.py: 在板端调起硬件加速器并进行推理测试的 Python 驱动脚本。

### 2. Doc (项目文档与参考资料)

包含项目相关的学术论文、技术报告及分工说明。

* FINN_2017/2018.pdf: FINN 框架核心参考文献。
* 基于_FINN_框架和_HLS_的量化神经网络FPGA实现.pdf: 本项目的中文技术报告。
* 其他关于 LeNet-5 优化和量化神经网络快速原型的参考资料。

### 3. figure (运行结果与流程截图)

存放训练过程及 FINN 编译各阶段的可视化结果。

* Dataflow_parent.png / dataflow.png: 硬件数据流图。
* streamline_1.png / tidy_after.png: FINN 转换层级时的中间结构图。
* top_sim1.png: 硬件推理验证与仿真波形图。
* 部署准确率验证.png / 环境部署.png: 实际运行测试截图。

### 4. models (ONNX 模型产物)

存放 FINN 编译器在不同转换阶段生成的中间 ONNX 模型，记录了模型从拓扑结构到硬件映射的变化。

* lenet_w2a1: 原始导出的量化模型。
* lenet_streamlined.onnx: 经过层间精简优化后的模型。
* lenet_folded.onnx: 经过折叠（资源分配）后的模型。
* lenet_dataflow_model.onnx: 最终生成数据流结构的硬件模型。
* lenet_synth.onnx: 综合后的硬件映射模型。

### 5. notebooks (FINN 编译流程)

存放用于交互式编译和验证的 Jupyter Notebook。

* lenet_end2end_example.ipynb: 从 ONNX 到比特流生成的完整端到端流程记录。
* lenet_end2end2_verification.ipynb: 针对硬件生成结果的验证与精度测试逻辑。

### 6. training (核心训练源码)

包含量化感知训练的所有逻辑。

* Letnet.py: 定义量化网络结构（权重2bit/激活1bit等）并进行 PyTorch 训练的脚本。
* training_curves.png: 训练过程中记录的准确率与损失值变化曲线。

### 7. 根目录文件

* .gitignore: Git 忽略清单，已配置忽略 build 产生的临时文件。
* Readme.md: 本项目总览说明文档。

---

## 技术概要

* 硬件平台: 针对 Xilinx Zynq-7000 系列 FPGA (PYNQ-Z2) 进行优化。
* 工具链: PyTorch, Brevitas, FINN, Vivado/Vitis HLS 2022.2，需要在FINN的Docker中运行。

