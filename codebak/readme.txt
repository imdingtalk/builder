main.py为主程序，也是唯一程序，里面首先定义了必要的子函数，最后定义了main函数，并调用该函数；
parameter_file文件夹中包含以下参数文件：
acIlmdb_vocab.npy：为20000个词的词表，用于字符到唯一索引的变换；
checkpoint_best.pth：为模型的权重参数

使用说明
--------

此镜像包含用于文本分类的模型。

启动命令:
python main.py -r test_data.txt -o result.log

启动时，脚本将自动安装三方库

参数说明:
-r : 指定测试数据文件路径 (test_data.txt)
-o : 指定输出结果文件路径 (result.log)

输出格式:
每条结果将包含以下字段:
{ID:测试样本id, Label:类别标签, prob:概率}
类别标签与概率均为字符串类型。

