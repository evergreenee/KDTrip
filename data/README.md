# preprocess.py文件中现存的问题（不影响功能）

## 函数get_data()的bug
- 传入的参数TRAIN_TRA实际上没有用，目前只在K = len(TRAIN_TRA)用来得到该数据集的序列条数，但实际上用K = len(new_trainTs)也是一样的结果。
- input_time_ = pad_time_batch(input_time)，在此处对一个batch的时间序列数据进行pad填充，使得一个batch中所有序列长度一致。但是这里填充的数据是‘0’，而0在原始序列中是有实际意义的，即表示0：00-1：00am，这会导致pad时引入噪声信息。
- 在处理生成decoder_batch时，不懂为什么要在每个序列末尾加上‘END’？
    - 经过实验测试后发现，删除‘END’的实验结果在F1分数上效果有0.02的下降，pairs-F1分数上效果有0.03的上升。所以改不改差别都不大。
    - 实验代码在“code4try/try.py”
- 这个函数的主要处理就是将所有序列数据按照序列长度进行升序排序，并对同处一个batch的数据进行pad补全，最终得到用于训练阶段的POI input序列、ground_truth（末尾加上END）、时间序列、据起点距离序列、据终点距离序列、序列长度等数据。
- 虽然说在问题定义里，起始POI和终止POI是给定的，只需要输出中间的POI点，但是在模型decoder部分的实际处理中，decoder是根据‘GO’token以及一些条件信息生成包括起终点在内的所有POI点的，只是在计算F1指标时，手动将序列的起止点更换成了label而已（提高F1分数的trick）。后续可以考虑尝试在不替换的情况下，原DeepTrip和我的模型的表现效果的对比？