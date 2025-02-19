# preprocess the original dataset(csv)
# if you want to run DeepTrip use other travel datasets, you just need make your datafile's format like these original data,and then run this python file
# After Preprocess, you can get 

from utils.util import *
import time
import math
import pickle
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler

def extract_words_vocab(voc_poi):
    int_to_vocab = {idx: word for idx, word in enumerate(voc_poi)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}
    return int_to_vocab, vocab_to_int

def extract_trajectory_features(df):
    features = []
    for traj_id, group in df.groupby('trajID'):
        start_poi = group.iloc[0]
        end_poi = group.iloc[-1]
        
        startPOI_ID = start_poi['poiID']
        startPOI_Cat = start_poi['poiCat']
        startPOI_time = start_poi['startPOI_time']
        
        endPOI_ID = end_poi['poiID']
        endPOI_Cat = end_poi['poiCat']
        endPOI_time = end_poi['startPOI_time']
        
        distance_to_end = start_poi['distance_to_end']

        traj_len = start_poi['trajLen']
        
        features.append({
            'startPOI_ID': startPOI_ID,
            'startPOI_Cat': startPOI_Cat,
            'startPOI_time': startPOI_time,
            'endPOI_ID': endPOI_ID,
            'endPOI_Cat': endPOI_Cat,
            'endPOI_time': endPOI_time,
            'traj_len': traj_len,
            'distance_to_end': distance_to_end,
            'trend_label': start_poi['trend_label'] 
        })
    return pd.DataFrame(features)

def processOriData(city_name, indice, batch_size):
    #load original data
    op_tdata = open('/root/KDTrip/dataset/origin_data/poi-'+ city_name + '.csv', 'r')
    ot_tdata = open('/root/KDTrip/dataset/origin_data/traj-'+ city_name + '-order.csv', 'r')
    trend_data = pd.read_csv('/root/KDTrip/dataset/dataAnaly/disAnaly/trajectory_distances_'+ city_name + '_test.csv')
    # process extra trajectory trend data
    trend_data['startTime'] = pd.to_datetime(trend_data['startTime'])
    trend_data['startPOI_time'] = (trend_data['startTime'] - trend_data['startTime'].min()).dt.total_seconds()  
    trend_features = extract_trajectory_features(trend_data)
    le_poi_cat = LabelEncoder()
    trend_features['endPOI_Cat'] = le_poi_cat.fit_transform(trend_features['endPOI_Cat'])
    trend_features['startPOI_Cat'] = le_poi_cat.transform(trend_features['startPOI_Cat'])
    le_trend_label = LabelEncoder()
    trend_features['trend_label'] = le_trend_label.fit_transform(trend_features['trend_label'])
    scaler = RobustScaler()
    trend_features_scaled = scaler.fit_transform(trend_features[['startPOI_time', 'endPOI_time', 'distance_to_end']])

    # load POI from csv file
    POIs=[]
    Trajectory=[]
    for line in op_tdata.readlines():
        lineArr = line.split(',')
        temp_line=list()
        for item in lineArr:
            temp_line.append(item.strip('\n'))
        POIs.append(temp_line)
    POIs=POIs[1:] #remove first line
    # get POIs and these longtitude/latitude
    get_POIs={}
    char_pois=[] #pois chars
    for items in POIs:
        char_pois.append(items[0])
        get_POIs.setdefault(items[0],[]).append([items[2],items[3]]) # pois to category
    # load Traj from csv file and calculate the visit frequency of POIs
    Users=[]
    poi_count={}
    for line in ot_tdata.readlines():
        lineArr=line.split(',')
        temp_line=list()
        if lineArr[0]=='userID':
            continue
        poi_count.setdefault(lineArr[2], []).append(lineArr[2])
        for i in range(len(lineArr)):
            if i==0:
                user = lineArr[i]
                Users.append(user)  # add user id
                temp_line.append(user)
                continue
            temp_line.append(lineArr[i].strip('\n'))
        Trajectory.append(temp_line)
    Users=sorted(list(set(Users)))
    print('user number',len(Users))
    # get reindiced trajectory sequence and corresponding user、time、distance sequence
    TRAIN_TRA=[]
    TRAIN_USER=[]
    TRAIN_TIME=[]
    TRAIN_DIST=[]
    DATA={} #temp_data
    for index in range(len(Trajectory)):
        if(int(Trajectory[index][-2])>=3): #the length of the trajectory must over than 3
            DATA.setdefault(Trajectory[index][0]+'-'+Trajectory[index][1],[]).append([Trajectory[index][2],Trajectory[index][3],Trajectory[index][4]]) #userID+trajID
    #calc_distance
    distance_count=[]
    for key in DATA.keys():
        traj=DATA[key]
        #print traj
        for i in range(len(traj)):
            #print get_POIs[traj[i][0]][0][0]
            lon1=float(get_POIs[traj[i][0]][0][0])
            lat1=float(get_POIs[traj[i][0]][0][1])
            for j in range(i+1,len(traj)):
                lon2 = float(get_POIs[traj[j][0]][0][0])
                lat2 = float(get_POIs[traj[j][0]][0][1])
                distance_count.append(calc_dist_vec(lon1,lat1,lon2,lat2))
    max_dis=max(distance_count)
    lower_dis=min(distance_count)
    print(len(distance_count))
    for keys in DATA.keys():
        user_traj=DATA[keys]
        temp_poi=[]
        temp_time=[]
        temp_dist=[]
        for i in range(len(user_traj)):
            temp_poi.append(user_traj[i][0]) #add poi id
            lon1=float(get_POIs[user_traj[i][0]][0][0])
            lat1=float(get_POIs[user_traj[i][0]][0][1])
            lons=float(get_POIs[user_traj[0][0]][0][0])
            lats=float(get_POIs[user_traj[0][0]][0][1])
            lone=float(get_POIs[user_traj[-1][0]][0][0])
            late=float(get_POIs[user_traj[-1][0]][0][1])
            sd = calc_dist_vec(lon1,lat1,lons,lats)
            ed = calc_dist_vec(lon1, lat1, lone, late)
            value1=0.5*(sd)/max(distance_count)
            value2=0.5*(ed)/max(distance_count)
            #print value
            temp_dist.append([value1,value2]) #lon,lat
            # trans time to hour
            dt = time.strftime("%H:%M:%S", time.localtime(int(user_traj[i][1:][0])))
            #print dt.split(":")[0]
            temp_time.append(int(dt.split(":")[0])) #add poi time
        TRAIN_USER.append(keys)
        TRAIN_TRA.append(temp_poi)
        TRAIN_TIME.append(temp_time)
        TRAIN_DIST.append(temp_dist)
    dictionary={}
    for key in poi_count.keys():
        count=len(poi_count[key])
        dictionary[key]=count
    max_count = max(dictionary.values())    #最大访问频次
    dictionary['GO']= max_count+1   #使重新编码后1表示‘GO’ID，0表示‘PAD’或‘MASK’ID
    dictionary['PAD']= max_count+2
    dictionary['END']=1
    '''
    dictionary['GO']=1
    dictionary['PAD']=1
    dictionary['END']=1
    '''
    # Sort POI IDs by visit frequency
    new_dict=sorted(dictionary.items(),key = lambda x:x[1],reverse = True)

    print('poi number is',len(new_dict)-3)
    voc_poi=list()

    # 按照访问频次重新设置POI ID,例如原“20”POI访问频次最高，则将该POI ID置为0
    for item in new_dict:
        voc_poi.append(item[0]) #has been sorted by frequency
    int_to_vocab, vocab_to_int=extract_words_vocab(voc_poi)
    print(int_to_vocab)
    print(vocab_to_int)

    #generate pre-traning dataset
    new_trainT = list()
    for i in range(len(TRAIN_TRA)): #TRAIN
        temp = list()
        temp.append(vocab_to_int['GO'])
        for j in range(len(TRAIN_TRA[i])):
            temp.append(vocab_to_int[TRAIN_TRA[i][j]])
        temp.append(vocab_to_int['END'])
        temp.append(vocab_to_int['PAD'])
        new_trainT.append(temp)

    #generate traning dataset
    new_trainTs = list()
    for i in range(len(TRAIN_TRA)): #TRAIN
        temp = list()
        for j in range(len(TRAIN_TRA[i])):
            temp.append(vocab_to_int[TRAIN_TRA[i][j]])
        new_trainTs.append(temp)
    # save POI sequence dataset
    dataset=open('./dataset/data/'+city_name+'_set.dat','w')
    for i in range(len(new_trainTs)):
        dataset.write(str(TRAIN_USER[i])+'\t')
        for j in range(len(new_trainTs[i])):
            dataset.write(str(new_trainTs[i][j])+'\t')
        dataset.write('\n')
    dataset.close()
    # Save vocab_to_int to a file
    with open('./dataset/vocab/vocab_to_int_'+city_name+'.pkl', 'wb') as f:
        pickle.dump(vocab_to_int, f)
    # Save to a file
    with open('./dataset/data/'+city_name+'_max_distance.json', 'w') as f:
        json.dump({"max_distance": max(distance_count)}, f, ensure_ascii=False, indent=4)
    # Save reindex POI ID and these longitutes/latitudes to a file
    poi_id_latlon = {}
    for item in new_dict:
        poi = item[0]
        if poi in get_POIs:
            lon, lat = get_POIs[poi][0]
            poi_id_latlon[vocab_to_int[poi]] = [str(lon), str(lat)]
    poi_id_latlon_file = f'./dataset/data/{city_name}_poi_id_latlon.json'
    with open(poi_id_latlon_file, 'w', encoding='utf-8') as f:
        json.dump(poi_id_latlon, f, ensure_ascii=False, indent=4)

    # map POI ID in trajectory trend data to int
    trend_features['startPOI_ID'] = trend_features['startPOI_ID'].astype(str).map(vocab_to_int)
    trend_features['endPOI_ID'] = trend_features['endPOI_ID'].astype(str).map(vocab_to_int)
    trend_features = np.hstack((trend_features_scaled, trend_features[['startPOI_ID', 'startPOI_Cat', 'endPOI_ID', 'endPOI_Cat', 'traj_len', 'trend_label']].values))
    
    return get_data(indice, TRAIN_TRA, new_trainTs, TRAIN_USER, TRAIN_TIME, TRAIN_DIST, batch_size, vocab_to_int, trend_features)


def get_data(index, TRAIN_TRA, new_trainTs, TRAIN_USER, TRAIN_TIME, TRAIN_DIST, batch_size, vocab_to_int, trend_features):
    # parameter 'batch_size' is used for completing the batch, if don't want this parameter, can use the parameter in Pytorch'DataLoader which is named 'drop_last' to control it
    #将原始数据划分为训练集和测试集，每次划分都取value个数据作为测试集，其余的数据作为训练集；
    #如果index等于0，则将排序后的前value个数据作为test，index等于K-1，则将后value个数据作为test；
    #否则，则取index*value~(index+1)*value的数据作为test
    #TRAIN_TRA表示原始POI ID 序列数据（未加PAD等额外token）,new_trainTs表示对POI ID按照访问频次排序后重新编码的POI序列
    #trend_features表示
    # sort original data
    K = len(TRAIN_TRA)
    print('traj number is', K)
    index_T = {}    #key：第i个序列，value：序列长度
    trainT = []     #从new_trainTs中切片选取训练数据，且序列按照序列长度升序排列
    trainU = []     #对应trainT序列中的userid
    trainTime=[]
    trainDist=[]
    trainFeature = []
    for i in range(len(new_trainTs)):
        index_T[i] = len(new_trainTs[i])
    temp_size = sorted(index_T.items(), key=lambda item: item[1])   #将字典 index_T 按值进行排序的操作，然后将结果赋值给变量 temp_size
    for i in range(len(temp_size)):
        id = temp_size[i][0]
        trainT.append(new_trainTs[id])
        trainU.append(TRAIN_USER[id])
        trainTime.append(TRAIN_TIME[id])
        trainDist.append(TRAIN_DIST[id])
        trainFeature.append(trend_features[id]) #trend_features.iloc[id].values
    
    value = int(K * 0.2)   # Step 1: Calculate 20% of len
    if value % batch_size != 0:
        value = (value // batch_size) * batch_size
    if index==K-1:
        testT=trainT[-value:]
        testU=trainU[-value:]
        trainT=trainT[:-value]
        trainU=trainU[:-value]

        testTime=trainTime[-value:]
        testDist=trainDist[-value:]
        trainTime=trainTime[:-value]
        trainDist=trainDist[:-value]

        testFeature = trainFeature[-value:]
        trainFeature = trainFeature[:-value]

    elif index==0:
        testT=trainT[:(index+1)*value]
        testU=trainU[:(index+1)*value]
        trainT =trainT[(index+1)*value:]
        trainU =trainU[(index+1)*value:]

        testTime=trainTime[:(index+1)*value]
        testDist=trainDist[:(index+1)*value]
        trainTime=trainTime[(index+1)*value:]
        trainDist=trainDist[(index+1)*value:]

        testFeature = trainFeature[:(index+1)*value]
        trainFeature = trainFeature[(index+1)*value:]

    else:
        testT=trainT[index*value:(index+1)*value]
        testU=trainU[index*value:(index+1)*value]
        trainT = trainT[0:index*value]+trainT[(index+1)*value:]
        trainU = trainU[0:index*value]+trainU[(index+1)*value:]

        testTime=trainTime[index*value:(index+1)*value]
        testDist=trainDist[index*value:(index+1)*value]
        trainTime=trainTime[0:index*value]+trainTime[(index+1)*value:]
        trainDist=trainDist[0:index*value]+trainDist[(index+1)*value:]

        testFeature = trainFeature[index*value:(index+1)*value]
        trainFeature = trainFeature[0:index*value]+trainFeature[(index+1)*value:]
        
    train_size = len(trainT) % batch_size
    if train_size != 0:
        trainT = trainT + [trainT[-1]]*(batch_size-train_size)  # copy data and fill the last batch size
        trainU = trainU + [trainU[-1]]*(batch_size-train_size)
        trainTime=trainTime+[trainTime[-1]]*(batch_size-train_size)
        trainDist = trainDist + [trainDist[-1]] * (batch_size - train_size)
        trainFeature = trainFeature + [trainFeature[-1]] * (batch_size - train_size)
    
    test_size = len(testT) % batch_size     #对测试数据进行batch补全
    if test_size!=0:
        testT = testT + [testT[-1]]*(batch_size-test_size)  # copy data and fill the last batch size
        testU = testU + [testU[-1]]*(batch_size-test_size)  #BUG for test_size<batch_size len(train_size<test_size)
        testTime=testTime+[testTime[-1]]*(batch_size-test_size)
        testDist = testDist + [testDist[-1]] * (batch_size - test_size)
        testFeature = testFeature + [testFeature[-1]] * (batch_size - test_size)
    print('test size',test_size,len(testT))
    
    #pre-processing
    #对数据预处理
    step=0
    encoder_train=[]    #添加了“PAD"作为序列补全的trainT
    decoder_trian=[]    #添加了”PAD"和“END”作为序列补全的trainT
    encoder_test=[]
    decoder_test=[]
    n_trainTime=[]  #做了序列补全的trainTime
    n_testTime=[]
    n_trainDist1=[]
    n_trainDist2= []
    n_testDist1=[]
    n_testDist2= []
    train_batch_lenth=[]    #训练集中每一个序列的长度
    test_batch_lenth=[]     #测试集中每一个序列的长度
    z_train=[]      #训练集中每一个序列的首尾POI
    z_train_time=[] #训练集中每一个序列的首尾POI的访问时间
    z_train_dist1=[]
    z_train_dist2 = []
    z_test=[]
    z_test_time=[]
    z_test_dist1=[]
    z_test_dist2 =[]
    while step < len(trainU) // batch_size:
        #每个batch处理一次
        start_i = step * batch_size
        input_x = trainT[start_i:start_i + batch_size]  #从trainT中选取一个batch的数据
        #time
        input_time = trainTime[start_i:start_i + batch_size]
        input_time_ = pad_time_batch(input_time)
        input_d = trainDist[start_i:start_i + batch_size]
        #input
        encode_batch = pad_sentence_batch(input_x, vocab_to_int['PAD'])
        decode_batchs = []
        z_batch=[]
        z_batch_time=[]
        z_batch_dist1=[]
        z_batch_dist2=[]
        for sampe in input_x:
            value = sampe
            value_=[sampe[0],sampe[-1]] #取一个序列的首尾POI
            decode_batchs.append(value)
            z_batch.append(value_)
        for sample in input_time:
            z_batch_time.append([sample[0],sample[-1]])
        decode_batch_ = eos_sentence_batch(decode_batchs, vocab_to_int['END'])
        #decode_batch_ = decode_batchs
        decode_batch = pad_sentence_batch(decode_batch_, vocab_to_int['PAD'])
        

        dist_1 = []
        dist_2 = []
        # print 'value',input_d
        for i in range(len(input_d)):
            temp_dist1 = []
            temp_dist2 = []
            for j in range(len(input_d[i])):
                temp_dist1.append(input_d[i][j][0])
                temp_dist2.append(input_d[i][j][1])
            dist_1.append(temp_dist1)
            dist_2.append(temp_dist2)
            z_batch_dist1.append([temp_dist1[0],temp_dist1[-1]])
            z_batch_dist2.append([temp_dist2[0], temp_dist2[-1]])
        dist_1_ = pad_dist_batch(dist_1)
        dist_2_ = pad_dist_batch((dist_2))

        pad_source_lengths = []
        for source in decode_batchs:
            pad_source_lengths.append(len(source) + 1)
            #pad_source_lengths.append(len(source))
        for i in range(batch_size):
            encoder_train.append(encode_batch[i])
            decoder_trian.append(decode_batch[i])
            train_batch_lenth.append(pad_source_lengths[i])
            n_trainTime.append(input_time_[i])
            n_trainDist1.append(dist_1_[i])
            n_trainDist2.append(dist_2_[i])
            z_train.append(z_batch[i])
            z_train_time.append(z_batch_time[i])
            z_train_dist1.append(z_batch_dist1[i])
            z_train_dist2.append(z_batch_dist2[i])
        step+=1
        #append to
    
    steps=0
    while steps < len(testU) // batch_size:
        start_i = steps * batch_size
        input_x = testT[start_i:start_i + batch_size]
        # time
        input_time = testTime[start_i:start_i + batch_size]
        input_time_ = pad_time_batch(input_time)
        input_d = testDist[start_i:start_i + batch_size]
        # input
        encode_batch = pad_sentence_batch(input_x, vocab_to_int['PAD'])
        decode_batchs = []
        z_batch = []
        z_batch_time = []
        z_batch_dist1 = []
        z_batch_dist2 = []
        for sampe in input_x:
            value = sampe
            value_ = [sampe[0], sampe[-1]]
            decode_batchs.append(value)
            z_batch.append(value_)
        for sample in input_time:
            z_batch_time.append([sample[0], sample[-1]])
        # ???可以考虑不加‘END’试试
        decode_batch_ = eos_sentence_batch(decode_batchs, vocab_to_int['END'])
        #decode_batch_ = decode_batchs
        decode_batch = pad_sentence_batch(decode_batch_, vocab_to_int['PAD'])
        

        dist_1 = []
        dist_2 = []
        # print 'value',input_d
        for i in range(len(input_d)):
            temp_dist1 = []
            temp_dist2 = []
            for j in range(len(input_d[i])):
                temp_dist1.append(input_d[i][j][0])
                temp_dist2.append(input_d[i][j][1])
            dist_1.append(temp_dist1)
            dist_2.append(temp_dist2)
            z_batch_dist1.append([temp_dist1[0], temp_dist1[-1]])
            z_batch_dist2.append([temp_dist2[0], temp_dist2[-1]])
        dist_1_ = pad_dist_batch(dist_1)
        dist_2_ = pad_dist_batch((dist_2))

        pad_source_lengths = []
        for source in decode_batchs:
            pad_source_lengths.append(len(source) + 1)
            #pad_source_lengths.append(len(source))

        for i in range(batch_size):
            encoder_test.append(encode_batch[i])
            decoder_test.append(decode_batch[i])
            test_batch_lenth.append(pad_source_lengths[i])
            n_testTime.append(input_time_[i])
            n_testDist1.append(dist_1_[i])
            n_testDist2.append(dist_2_[i])
            z_test.append(z_batch[i])
            z_test_time.append(z_batch_time[i])
            z_test_dist1.append(z_batch_dist1[i])
            z_test_dist2.append(z_batch_dist2[i])
        steps+=1
        

    #deal trend data
    trainFeature = np.array(trainFeature)
    testFeature = np.array(testFeature)   #testFeature = np.array(testFeature)
    train_trend_label = trainFeature[:, -1].astype(int)
    test_trend_label = testFeature[:, -1].astype(int)
    train_features = trainFeature[:, :-1]
    test_features = testFeature[:, :-1]

    train_variables = [encoder_train,decoder_trian,train_batch_lenth, n_trainTime,n_trainDist1, n_trainDist2,z_train,z_train_time,z_train_dist1,z_train_dist2,train_features,train_trend_label]
    test_variables = [encoder_test, decoder_test, test_batch_lenth, n_testTime, n_testDist1, n_testDist2, z_test, z_test_time, z_test_dist1, z_test_dist2,test_features,test_trend_label]
    return train_variables,test_variables

