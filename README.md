# trtFaceReco




### 1. 流程

使用TensorRT加速人脸识别。
整个流程可以分为两个环节，a) 存储人脸特征 b) 人脸识别

a) 读取图像中的人脸信息 > 提取人脸特征 > 将提取到的人脸特征存储到csv文件内  
b) 读取图像中的人脸信息 > 提取人脸特征 > 未知人脸与已知人脸作对比  

model路径下共有5个模型  

a) retinaFace.onnx onnx版本的人脸检测建模型
   retinaFace.trt  由onnx版本转换得到的TensorRT模型  
<font color="Red">注意！ 如果在不同的硬件、不同的TensorRT版本上运行，请重新使用trtexec转换 </font>  

b) facenet_op.onnx onnx版本的人脸特征表示模型。
   facenet_op10.onnx 以上2个模型都是由pytroch转换为onnx的模型，不同点是使用了不同opset版本，
在使用onnx模型转换为TensorRT模型时，如果某个不能用，可以尝试使用另外一个  
   facenet_op10.trt 使用facenet_op10.onnx转换而来的TensorRT模型
 
<font color=red>注意！ 以上模型均已测试可以正常运行。</font>

### 2. 存储人脸特征

使用 getFaceFromCamera.py 文件提取人脸特征信息

画面中只有一个人脸的时候才会做后续处理即提取相应的人脸特征信息

使用retinaface和facenet两个模型实现效果，两个模型存储路径为model

存储的人脸特征信息会放在data/feature.csv文件下

按'n'键新建人脸信息，程序会自动在data路径下新建一个文件夹存储即将要保存的人脸图像  
按's'键保存当前图像页面中检测到的人脸并存储在data路径下的某个文件夹内  
按'q'键退出界面

### 3. 人脸识别

整个人脸识别过程遵循通用的流程，即人脸检测->人脸配准（待做）->人脸表示。  
使用faceRecoFromCamera.py，会将图像中所有检测到的人脸信息与feature.csv中暂存的信息作对比，  
计算两个人脸之间的欧式距离，距离越小代表这两张人脸越相似，会在代码中预设一个阈值，建议阈值为0.6。