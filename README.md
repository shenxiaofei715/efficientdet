# efficientdet
Train your own dataset
1.标注自己的数据集
  标注工具使用标注精灵软件,因项目需要本人标注数据格式是标注四边形(四个位置点(x,y))
2.转换自己的数据集为tfrecord文件
   打开dataset文件夹中的create_tfrecord.py文件 修改一下内容
      NAME_LABEL_MAP = {'back_ground': 0,
                         '焊点': 1,
                         '标签': 2 }
   运行dataset文件夹中的create_tfrecord.py文件
  
  
