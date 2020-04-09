# efficientdet
Train your own dataset

1.标注自己的数据集
  标注工具使用标注精灵软件,因项目需要本人标注数据格式是标注四边形(四个位置点(x,y))

2.转换自己的数据集为tfrecord文件

   2.1打开dataset文件夹中的create_tfrecord.py文件 修改一下内容
      修改:NAME_LABEL_MAP = {'back_ground': 0,
                             '焊点': 1,
                             '标签': 2 }
                        
          建立自己的数据集中识别目标类别名称和class_id之间的映射,例如标注过程中识别目标是焊点和标签
          
      修改函数参数create_tf_record_from_xml(image_path=保存训练图像的文件夹路径,
                              xml_path  =保存训练图像的文件夹路径xml文件的文件夹路径
                              tf_output_path=生成的tfrecord的文件保存文件夹路径,
                              tf_record_num_shards=5,img_format=图像的格式(".jpg/png"))  
                              
        
   2.2 运行dataset文件夹中的create_tfrecord.py文件即可完成数据集转换为tfrecord文件
  
  
3.训练自己的数据集
    3.1打开hparams_config.py文件修改其中与数据集配置相关的默认参数
                 # dataset specific parameter
                 h.num_classes =2----数据集的识别类别数目
                 
    3.2打开main.py修改一下配置参数
        
       flags.DEFINE_string('model_dir', "训练模型的保存文件夹路径", 'Location of model_dir')
       flags.DEFINE_integer('train_batch_size',制定训练bach的大小, 'training batch size')
       
      flags.DEFINE_string('training_file_pattern', "tfrecord文件保存的1文件夹路径/*.tfrecord",
      'Glob for training data files (e.g., COCO train - minival set)')
      
       flags.DEFINE_integer('num_examples_per_epoch',训练数据的example个数,
                     'Number of examples in one epoch')
       flags.DEFINE_integer('num_epochs',训练的迭代次数, 'Number of epochs for training')
       flags.DEFINE_string('model_name', '指定模型名称例如efficientdet-d0',
                    'Model name: retinanet or efficientdet')
    
    3.3运行main.py即可开始训练自己的数据集的
                    
  
        

