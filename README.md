"This project encompasses the primary code from my graduation design. Upon obtaining the RetinalOCT-C8dataset that I utilized, you can fully reproduce the experimental content of the paper."

“这个项目包含了我的主要毕业设计代码内容，找到和我一样的RetinalOCT-C8数据集之后可以完整复刻论文内容实验”

Model里面的代码是五种基本的模型。其中EfiicientNet和Swin Transformer有多种变体，可以再model_dict里面查看初始化名称;

Train脚本是进行模型对比的时候训练的基本脚本。基本参数可以调。

F1Pro脚本是模型评估脚本，评估的脚本选择数据集需要选择到整个数据集的测试集，也就是test，不能是训练集和验证集，否则出现谬误。脚本计算如F1分数，Precision和 recall的，同时支持宏和加权两种模式的分数，根据数据集的不同加权分数和宏分数会出现不同，本实验基于RetinalOCT-C8数据集，由于各分类数目都是3000张图片，数目相同，所以加权分数等于宏分数。

本论文数据集在https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8下载。

其他如Adam.py、AdamW.py、AMP.py、CosineAnnealingLR.py、RMSProp.py. smooth.py、smooth2.py、StepLR.py是消融试验的变量训练器。是基于Train.py修改得来(Train.py是论文中的A1变量)。可以看论文控制变量表格。 Finally.py是最终模型训练的脚本,经过消融试验最后的训练器。

system.py是医学图像分类系统脚本。分类脚本不要对拿来训练的训练集和测试集的图片分类。

由于实验环境是租的AutoDL云GPU，整体代码的运行相关逻辑需要修改指定地址。比如Train训练Resnet的时候，具体模型地址和数据集地址需要依据环境决定

图像分类系统代码是system.py,可以直接运行，注意不能在虚拟环境，最好在本地环境，不然运行可能报错。

另外诸多论文图片的绘制代码删了一部分，但保留了训练曲线绘制，如Draw.py。

同时训练曲线也可以用tensorboard功能绘制。每个训练器脚本训练完会有一个事件文件例如:events.out.tfevents.1747599759.autodl-container-a09f4e8ad1-cca672ef，这个可以用tensorboard绘制。具体方法不详细讲。另外也有一个txt文件，完整记录了训练日志。可以用Draw.py绘制漂亮的曲线，包括训练曲线、损失曲线、学习率变化曲线，其中训练曲线和损失曲线又分为训练和验证两种。

另外对比试验和消融试验内存太大，没有放在github上面，实验过程的所有文件（包括各种模型权重文件，训练日志文件以及多种可视化图片以及tensorboardX的事件文件）都在百度网盘：通过网盘分享的文件：Result链接: https://pan.baidu.com/s/1pJf-t-VNEcVPC9KWkVDZLA 提取码: n69j
