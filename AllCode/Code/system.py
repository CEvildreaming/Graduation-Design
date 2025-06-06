# -*- coding: utf-8 -*-

"""
医学图像分类测试系统
第一次调试功能5.13

第二次调试函数映射按钮

第三次调试自动检测类别功能

第四次调试样本库功能
功能：
1. 选择预训练模型
2. 加载图片进行分类
3. 显示分类结果和概率分布
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGroupBox, QFormLayout, 
                             QLabel, QLineEdit, QPushButton, QSpinBox,
                             QComboBox, QFileDialog, QTextEdit,
                             QDialogButtonBox, QDialog, QListWidget, QMessageBox, QListWidgetItem)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QIcon, QPixmap

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号

class MedicalImageClassifierApp(QMainWindow):
    """OCT图像分类测试系统"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("医学图像分类测试系统")
        self.setGeometry(100, 100, 800, 600)
        
        self.model_choices = ["resnet18", "resnet50", "resnet101", "vit", "swin-t", "swin-s", "swin-b", 
                             "googlenet", "efficientnet-b0", "efficientnet-b1", "efficientnet-b2"]
        
        # 保存数据集路径
        self.dataset_path = ""
        # 尝试读取默认的Dataset目录
        if os.path.exists("Dataset"):
            self.dataset_path = "Dataset"
        
        self.init_ui()
        
    def init_ui(self):
        """初始化UI界面"""
        # 主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        central_widget.setStyleSheet("background-color:#f0f0f0;")
        
        # 设置分类测试页面
        self.setup_classification_ui(main_layout)
    
    def setup_classification_ui(self, layout):
        """设置图像分类测试UI"""
        # 创建模型选择和图像加载区域
        config_group = QGroupBox("模型与图像设置")
        config_layout = QFormLayout()
        
        # 数据集路径选择
        self.dataset_path_input = QLineEdit(self.dataset_path)
        dataset_browse_btn = QPushButton("浏览...")
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(self.dataset_path_input)
        dataset_layout.addWidget(dataset_browse_btn)
        config_layout.addRow("数据集路径:", dataset_layout)
        dataset_browse_btn.clicked.connect(self.browse_dataset)
        
        # 模型选择
        self.model_path_input = QLineEdit()
        model_browse_btn = QPushButton("浏览...")
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_path_input)
        model_layout.addWidget(model_browse_btn)
        config_layout.addRow("模型路径:", model_layout)
        model_browse_btn.clicked.connect(self.browse_model)
        
        # 图像选择
        self.image_path_input = QLineEdit()
        image_browse_btn = QPushButton("浏览...")
        validation_samples_btn = QPushButton("样本库...")
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.image_path_input)
        image_layout.addWidget(image_browse_btn)
        image_layout.addWidget(validation_samples_btn)
        config_layout.addRow("图像路径:", image_layout)
        image_browse_btn.clicked.connect(self.browse_image)
        validation_samples_btn.clicked.connect(self.browse_validation_samples)
        
        # 设置模型类型下拉框
        self.model_type_selector = QComboBox()
        self.model_type_selector.addItems(self.model_choices)
        config_layout.addRow("模型类型:", self.model_type_selector)
        
        # 设置类别数量
        self.classes_num = QSpinBox()
        self.classes_num.setRange(2, 100)
        self.classes_num.setValue(8) # 默认为8类，适合医学图像分类
        config_layout.addRow("类别数量:", self.classes_num)
        
        # 自动检测类别
        self.auto_detect_classes = QPushButton("自动检测类别")
        self.auto_detect_classes.clicked.connect(self.detect_classes)
        config_layout.addRow("", self.auto_detect_classes)
        
        config_group.setLayout(config_layout)
        
        # 验证按钮
        validate_btn = QPushButton("分析图像")
        validate_btn.clicked.connect(self.validate_image)
        validate_btn.setMinimumHeight(40)
        validate_btn.setStyleSheet("background-color:  #0000FF; color: white; font-weight: bold; font-size: 16px;")
        
        # 显示区域
        display_group = QGroupBox("分类结果")
        display_layout = QHBoxLayout()
        
        # 图像显示区域
        image_display_layout = QVBoxLayout()
        self.image_label = QLabel("图像预览")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(300, 300)
        self.image_label.setStyleSheet("border: 1px solid #cccccc;")
        image_display_layout.addWidget(self.image_label)
        
        # 结果显示区域
        result_display_layout = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        result_display_layout.addWidget(QLabel("分类结果:"))
        result_display_layout.addWidget(self.result_text)
        
        # 添加到显示布局
        display_layout.addLayout(image_display_layout, 1)
        display_layout.addLayout(result_display_layout, 1)
        display_group.setLayout(display_layout)
        
        # 添加到主布局
        layout.addWidget(config_group)
        layout.addWidget(validate_btn)
        layout.addWidget(display_group)
    
    def browse_dataset(self):
        """打开文件对话框选择数据集目录"""
        dir = QFileDialog.getExistingDirectory(self, "选择数据集目录", "")
        if dir:
            self.dataset_path_input.setText(dir)
            self.dataset_path = dir
            # 自动检测类别
            self.detect_classes()
    
    def detect_classes(self):
        """自动检测数据集中的类别及数量，并生成类别映射文件"""
        dataset_path = self.dataset_path_input.text()
        if not dataset_path or not os.path.exists(dataset_path):
            QMessageBox.warning(self, "警告", "请先选择有效的数据集路径")
            return
        
        # 首先尝试检查训练集目录
        train_dir = os.path.join(dataset_path, 'train')
        if os.path.exists(train_dir):
            class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
            if class_dirs:
                # 按字母顺序排序，确保与加载顺序一致
                class_dirs.sort()
                classes_count = len(class_dirs)
                
                # 更新类别数量
                self.classes_num.setValue(classes_count)
                
                # 显示检测到的类别
                class_info = f"检测到 {classes_count} 个类别（按字母顺序排序）:\n" + "\n".join(class_dirs)
                QMessageBox.information(self, "类别检测结果", class_info)
                
                # 将检测到的类别信息写入结果区域
                self.result_text.setText(f"已自动检测数据集类别：\n{class_info}\n\n请选择图像和模型进行分析。")
                
                # 生成类别映射文件，方便与模型对应
                try:
                    mapping_file = os.path.join(dataset_path, "classes.txt")
                    with open(mapping_file, 'w') as f:
                        for class_name in class_dirs:
                            f.write(f"{class_name}\n")
                    self.result_text.append(f"已将类别映射保存至: {mapping_file}")
                except Exception as e:
                    self.result_text.append(f"保存类别映射文件时出错: {str(e)}")
                
                return
            
        # 如果训练集目录不存在或为空，尝试验证集目录
        val_dir = os.path.join(dataset_path, 'val')
        if os.path.exists(val_dir):
            class_dirs = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
            if class_dirs:
                # 按字母顺序排序，确保与加载顺序一致
                class_dirs.sort()
                classes_count = len(class_dirs)
                
                # 更新类别数量
                self.classes_num.setValue(classes_count)
                
                # 显示检测到的类别
                class_info = f"检测到 {classes_count} 个类别（按字母顺序排序）:\n" + "\n".join(class_dirs)
                QMessageBox.information(self, "类别检测结果", class_info)
                
                # 将检测到的类别信息写入结果区域
                self.result_text.setText(f"已自动检测数据集类别：\n{class_info}\n\n请选择图像和模型进行分析。")
                
                # 生成类别映射文件，方便与模型对应
                try:
                    mapping_file = os.path.join(dataset_path, "classes.txt")
                    with open(mapping_file, 'w') as f:
                        for class_name in class_dirs:
                            f.write(f"{class_name}\n")
                    self.result_text.append(f"已将类别映射保存至: {mapping_file}")
                except Exception as e:
                    self.result_text.append(f"保存类别映射文件时出错: {str(e)}")
                
                return
                
        # 都找不到类别目录
        QMessageBox.warning(self, "警告", "无法在数据集中检测到类别目录，请确保数据集包含train或val子目录，且其中包含类别子目录")
    
    def browse_model(self):
        """打开文件对话框选择模型权重文件"""
        file, _ = QFileDialog.getOpenFileName(self, "选择模型权重文件", "", "权重文件 (*.pth);;所有文件 (*)")
        if file:
            self.model_path_input.setText(file)
    
    def browse_image(self):
        """打开文件对话框选择图像文件"""
        file, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)")
        if file:
            self.image_path_input.setText(file)
            # 预览图像
            pixmap = QPixmap(file)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(pixmap)
            else:
                self.image_label.setText("无法加载图像")

    def browse_validation_samples(self):
        """浏览并选择validation_samples文件夹中的样本图像"""
        # 确保validation_samples文件夹存在
        validation_dir = "./validation_samples"
        if not os.path.exists(validation_dir):
            os.makedirs(validation_dir, exist_ok=True)
        
        # 列出validation_samples文件夹中的图像
        sample_images = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            import glob
            sample_images.extend(glob.glob(os.path.join(validation_dir, ext)))
        
        if not sample_images:
            # 如果没有样本，提示用户
            QMessageBox.information(self, "提示", "样本库中没有图像，请先使用'浏览...'按钮添加图像到样本库")
            # 直接打开文件选择对话框添加图像
            self.add_more_samples(validation_dir)
            # 重新获取样本列表
            sample_images = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                sample_images.extend(glob.glob(os.path.join(validation_dir, ext)))
            
            if not sample_images:
                return
            
        # 创建样本选择对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("选择样本图像")
        dialog.setMinimumWidth(600)
        dialog.setMinimumHeight(400)
        
        # 创建列表视图
        list_widget = QListWidget()
        list_widget.setIconSize(QSize(64, 64))
        
        # 填充列表
        for img_path in sample_images:
            item = QListWidgetItem(os.path.basename(img_path))
            pixmap = QPixmap(img_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                item.setIcon(QIcon(pixmap))
            item.setData(Qt.UserRole, img_path)  # 存储完整路径
            list_widget.addItem(item)
        
        # 双击处理
        def on_item_double_clicked(item):
            path = item.data(Qt.UserRole)
            self.image_path_input.setText(path)
            # 预览图像
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(pixmap)
            dialog.accept()
        
        list_widget.itemDoubleClicked.connect(on_item_double_clicked)
        
        # 添加按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # 处理确认按钮
        def on_ok_clicked():
            selected_items = list_widget.selectedItems()
            if selected_items:
                path = selected_items[0].data(Qt.UserRole)
                self.image_path_input.setText(path)
                # 预览图像
                pixmap = QPixmap(path)
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.image_label.setPixmap(pixmap)
        
        button_box.accepted.disconnect(dialog.accept)
        button_box.accepted.connect(on_ok_clicked)
        button_box.accepted.connect(dialog.accept)
        
        # 添加"添加更多样本"按钮
        add_more_btn = QPushButton("添加更多样本...")
        add_more_btn.clicked.connect(lambda: self.add_more_samples(validation_dir, list_widget))
        
        # 布局
        layout = QVBoxLayout()
        layout.addWidget(QLabel("双击选择样本图像:"))
        layout.addWidget(list_widget)
        layout.addWidget(add_more_btn)
        layout.addWidget(button_box)
        dialog.setLayout(layout)
        
        # 显示对话框
        dialog.exec_()

    def add_more_samples(self, validation_dir, list_widget=None):
        """添加更多样本到validation_samples文件夹"""
        files, _ = QFileDialog.getOpenFileNames(
            self, 
            "选择样本图像", 
            "", 
            "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)"
        )
        
        if files:
            try:
                import shutil
                os.makedirs(validation_dir, exist_ok=True)
                
                for file_path in files:
                    # 复制文件到validation_samples文件夹
                    dst_file = os.path.join(validation_dir, os.path.basename(file_path))
                    shutil.copy2(file_path, dst_file)
                    
                    # 如果提供了列表控件，添加到列表中
                    if list_widget:
                        item = QListWidgetItem(os.path.basename(dst_file))
                        pixmap = QPixmap(dst_file)
                        if not pixmap.isNull():
                            pixmap = pixmap.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                            item.setIcon(QIcon(pixmap))
                        item.setData(Qt.UserRole, dst_file)
                        list_widget.addItem(item)
                
                QMessageBox.information(self, "成功", f"已添加 {len(files)} 个样本图像")
                return True
            except Exception as e:
                QMessageBox.critical(self, "错误", f"添加样本图像出错: {str(e)}")
                return False
        
        return False
    
    def get_class_names_from_dataset(self):
        """从数据集目录获取类别名称，确保按字母顺序排序以匹配训练模型时的顺序"""
        dataset_path = self.dataset_path_input.text()
        class_names = []
        
        # 首先检查是否有与模型路径相关的类别映射文件
        model_path = self.model_path_input.text()
        if model_path and os.path.exists(model_path):
            # 尝试查找与模型同名的类别映射文件
            model_dir = os.path.dirname(model_path)
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            
            # 检查几种可能的类别映射文件
            possible_mapping_files = [
                os.path.join(model_dir, f"{model_name}_classes.txt"),
                os.path.join(model_dir, "classes.txt"),
                os.path.join(model_dir, "class_mapping.txt"),
                os.path.join(model_dir, "class_names.txt"),
                os.path.join(os.path.dirname(model_dir), "classes.txt")
            ]
            
            for mapping_file in possible_mapping_files:
                if os.path.exists(mapping_file):
                    try:
                        with open(mapping_file, 'r') as f:
                            class_names = [line.strip() for line in f if line.strip()]
                        self.result_text.append(f"已从映射文件 {os.path.basename(mapping_file)} 读取 {len(class_names)} 个类别名称")
                        return class_names
                    except Exception as e:
                        self.result_text.append(f"读取类别映射文件时出错: {str(e)}")
        
        # 检查数据集目录中是否有类别映射文件
        if dataset_path and os.path.exists(dataset_path):
            classes_file = os.path.join(dataset_path, "classes.txt")
            if os.path.exists(classes_file):
                try:
                    with open(classes_file, 'r') as f:
                        class_names = [line.strip() for line in f if line.strip()]
                    self.result_text.append(f"已从数据集目录中的类别映射文件读取 {len(class_names)} 个类别")
                    return class_names
                except Exception as e:
                    self.result_text.append(f"读取数据集类别映射文件时出错: {str(e)}")
        
        # 如果没有映射文件，按照字母顺序排序目录
        # 首先尝试从训练集获取类别名称
        train_dir = os.path.join(dataset_path, 'train')
        if os.path.exists(train_dir):
            dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
            if dirs:
                # 按字母顺序排序，这通常与PyTorch的DataLoader加载顺序一致
                dirs.sort()
                return dirs
        
        # 如果训练集不存在或为空，尝试从验证集获取
        val_dir = os.path.join(dataset_path, 'val')
        if os.path.exists(val_dir):
            dirs = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
            if dirs:
                # 按字母顺序排序，这通常与PyTorch的DataLoader加载顺序一致
                dirs.sort()
                return dirs
        
        # 如果都没有找到，返回默认类别名
        class_count = self.classes_num.value()
        return [f"class{i+1}" for i in range(class_count)]

    def validate_image(self):
        """验证图像分类"""
        model_path = self.model_path_input.text()
        image_path = self.image_path_input.text()
        model_type = self.model_type_selector.currentText()
        classes_num = self.classes_num.value()
        
        if not model_path or not image_path:
            QMessageBox.warning(self, "警告", "请选择模型和图像文件")
            return
            
        if not os.path.exists(model_path) or not os.path.exists(image_path):
            QMessageBox.warning(self, "警告", "模型或图像文件不存在")
            return
        
        # 显示处理中提示
        self.result_text.setText("正在处理中...")
        QApplication.processEvents()  # 更新界面
        
        try:
            # 导入必要的库
            import torch
            from PIL import Image
            import torchvision.transforms as transforms
            from model import model_dict
            
            # 检查模型类型是否有效
            if model_type not in model_dict:
                self.result_text.setText(f"未知的模型类型: {model_type}\n可用的模型类型: {', '.join(list(model_dict.keys()))}")
                return
            
            # 根据模型类型选择适当的图像预处理
            transform = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize((0.2099, 0.2099, 0.2099), (0.1826, 0.1826, 0.1826))
            ])
            
            # 加载图像
            img = Image.open(image_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)  # 添加批次维度
            
            # 判断是否有GPU可用
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            img_tensor = img_tensor.to(device)
            
            # 加载模型权重文件
            checkpoint = torch.load(model_path, map_location=device)
            
            # 检查加载的权重文件类型
            if "generator" in checkpoint and "discriminator" in checkpoint:
                self.result_text.setText("您选择的是GAN模型权重文件，而不是分类模型权重文件。\n请选择训练好的分类模型权重文件。")
                return
            
            # 检查是否有类别映射信息保存在模型中
            class_names = []
            if "classes" in checkpoint:
                class_names = checkpoint["classes"]
                self.result_text.append(f"从模型文件中检测到 {len(class_names)} 个类别: {', '.join(class_names)}")
                classes_num = len(class_names)
                self.classes_num.setValue(classes_num)
            
            # 如果模型中没有类别信息，尝试从模型权重文件获取类别数量
            if not class_names:
                try:
                    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
                        # 查找分类层的权重以确定类别数量
                        for key in checkpoint["model"]:
                            if any(name in key for name in ["fc.weight", "classifier.weight", "head.weight"]):
                                weight_shape = checkpoint["model"][key].shape
                                if len(weight_shape) >= 1:
                                    detected_classes = weight_shape[0]
                                    if detected_classes != classes_num:
                                        classes_num = detected_classes
                                        self.classes_num.setValue(classes_num)
                                        self.result_text.append(f"从模型权重中检测到 {classes_num} 个类别，已自动更新")
                                    break
                except Exception as e:
                    self.result_text.append(f"检测模型类别数量时出错，使用当前设置值: {classes_num}\n错误: {str(e)}")
            
            # 加载分类模型
            model = model_dict[model_type](num_classes=classes_num, pretrained=False)
            model = model.to(device)
            
            # 尝试不同方式加载权重
            try:
                # 首先尝试标准格式：包含"model"键的字典
                if "model" in checkpoint:
                    model.load_state_dict(checkpoint["model"])
                # 然后尝试不同的状态字典格式
                elif "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                elif "weights" in checkpoint:
                    model.load_state_dict(checkpoint["weights"])
                # 对于不同模型结构的状态字典，需要更灵活的处理
                elif isinstance(checkpoint, dict):
                    # 尝试直接加载
                    try:
                        model.load_state_dict(checkpoint)
                    except Exception as e:
                        # 加载部分权重（跳过不匹配的层）
                        model_dict_weights = model.state_dict()
                        # 过滤掉不匹配的层
                        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict_weights and v.shape == model_dict_weights[k].shape}
                        if not pretrained_dict:
                            raise Exception("模型权重与当前模型结构不匹配，没有可加载的层")
                        # 更新模型状态字典并加载
                        model_dict_weights.update(pretrained_dict)
                        model.load_state_dict(model_dict_weights)
                    else:
                        self.result_text.setText(f"无法识别的模型权重格式。\n\n请确保选择了正确的分类模型权重文件。")
                        return
            except Exception as load_error:
                self.result_text.setText(f"加载模型权重出错:\n{str(load_error)}\n\n"
                                     f"请确保选择的模型类型（{model_type}）与权重文件匹配，并且类别数量设置正确。")
                return
            
            # 切换到评估模式
            model.eval()
            
            # 前向传播
            with torch.no_grad():
                outputs = model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze()
            
            # 获取结果
            predicted_class = predicted.item()
            confidence = probabilities[predicted_class].item() * 100
            
            # 如果从模型中没有获取到类别名称，从数据集获取
            if not class_names:
                class_names = self.get_class_names_from_dataset()
            
            # 如果类别数量不匹配，进行扩展或截断
            if len(class_names) < classes_num:
                # 扩展类别列表
                for i in range(len(class_names), classes_num):
                    class_names.append(f"class{i+1}")
            elif len(class_names) > classes_num:
                # 截断类别列表
                class_names = class_names[:classes_num]
            
            # 显示结果
            class_result = class_names[predicted_class] if predicted_class < len(class_names) else f"class{predicted_class+1}"
            result_str = f"预测类别: {class_result}\n置信度: {confidence:.2f}%\n\n类别概率分布:\n"
            
            # 显示所有类别的概率
            for i, prob in enumerate(probabilities):
                if i < classes_num:
                    class_name = class_names[i] if i < len(class_names) else f"class{i+1}"
                result_str += f"{class_name}: {prob.item()*100:.2f}%\n"
            
            self.result_text.setText(result_str)
            
        except Exception as e:
            self.result_text.setText(f"验证过程出错:\n{str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MedicalImageClassifierApp()
    window.show()
    sys.exit(app.exec_()) 