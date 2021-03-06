# ModifiedUnet

​		本方案所采用的模型的具体实现为 network.py 中的 ModifiedUnet。该模型在原有的 Unet 基础上增加了注意力机制，具体如下图所示。

<img src="/home/wang/Project/ZXPY/doc/模型.bmp" alt="模型"  />

​		图中蓝色模块表示原有的 Unet 结构，红色模块表示增加的注意力模块。考虑到原有 Unet 的不同特征层只描述了图像的局部特征，而图像的降噪是一个全局的问题，因此采用 CBAM (Convolutional Block Attention Module) 模块来添加注意力机制。CBAM 同时空间注意力和通道注意力，且即插即用，方便对模型进行修改。具体操作为：对模型的5个特征层分别插入CBAM模块，后续特征拼接和上采样不变。如此，每层的特征在不同尺度的特征表达之外，还包含通道间、空间的注意力信息，即添加了全局信息。

​		模型的训练采用迁移学习的思路，通过提供的 Unet 预训练模型对该模型中原有的 Unet 结构进行参数初始化，随后通过求和的 MSE 损失函数对模型进行训练，共 300 个 epoch，初始学习率设置为 0.001，并在 90 和 240 epoch 之后分别降低 0.5 倍。提供的 100 对图像中，80 对用于训练集，20对用于测试集。

​		模型最终在测试集上表现为：

| 指标 |  数值   |
| :--: | :-----: |
| PSNR | 46.3988 |
| SSIM | 0.99418 |

