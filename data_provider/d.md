我现在需要对我的代码进行一些重构，所有的dataset类都要继承自一个基类dataset_yoto. 这个类的功能是这样的，
首先有def __init__(self, data_type, seq_len, dataset_name, dataset_description, sampling_rate, dataset_train_or_val)，其中data type是只能2个值，分别是0 1 ， 0代表做分类，1代表做寿命预测，这个值作为参数给后面的_load_samples这个方法，这个方法会根据这个值来区分rul和故障分类的提取方法 seq_len表示length的长度比如1024，这个初始值设置为4096， 然后是dataset_name和dataset——description用来描述这个模块，samplingrate是数据集的采样率，比如xjtu的采样率是一分钟32768个点，就是32768/60的采样率。

接下来有一个self.fault_type_map = {
            'inner': [0, 1, 0, 0, 0],
            'outer': [0, 0, 1, 0, 0],
            'cage': [0, 0, 0, 1, 0],
            'ball': [0, 0, 0, 0, 1],
            'normal': [1, 0, 0, 0, 0]
        }
        然后是def getiiem和get __len__这两个工具函数，
        最后是def _load_samples这个类，这个只实现成接口，并接受data_type这个作为一个参数，并根据参数区分返回值。这里的返回值有两个，分别是
        x = [channel, length]
        if data_type = 0, y = [falut_type]
        elif data_type = 1, y = [rul_value]

然后，你要为每个数据集都实现一个继承自dataset_yoto类的一个dataset类，先从xjtu的开始实现，并在最后添加if __name__ == "__main__"的测试代码