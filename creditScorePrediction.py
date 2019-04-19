#coding = 'utf-8' 
from mxnet.gluon import data as gdata, loss as gloss, nn
# import d2lzh as d2l
import auxlib as aul
from mxnet import autograd, gluon, init, nd
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
# from tensorflow.metrics import mean_absolute_error
# mean_absolute_error(y_true,y_pred)
# loss = gloss.SoftmaxCrossEntropyLoss()
drop_prob1, drop_prob2, drop_prob3 = 0.2, 0.2, 0.1
#Model1 线性回归模型+平方损失函数
def get_net0():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net
loss = gloss.L2Loss()
def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(256, activation="relu"), nn.Dropout(drop_prob1), nn.Dense(512, activation="relu"), nn.Dropout(drop_prob2),
            nn.Dense(256, activation="relu"), nn.Dropout(drop_prob3), nn.Dense(1))
    # nn.Dense(256, activation="relu"), nn.Dropout(drop_prob2),  
    net.initialize()
    return net

#平方绝对误差
def calculateMSE(X,Y,m,b):
    in_bracket = []
    for i in range(len(X)):
        num = Y[i] - m*X[i] - b
        num = pow(num,2)
        in_bracket.append(num)   
    all_sum = sum(in_bracket)
    MSE = all_sum / len(X)
    return MSE
def calMAE(net, features, labels):
    clipped_preds = nd.clip(net(features), 1, float('inf'))
    mae_error = 0
    i = 0
    for element in (labels.log()-clipped_preds.log()):
        i += 1
        mae_error += element.abs()
    return (mae_error/i).asscalar()
def mae(net, y_true, y_pred):
    return np.mean(abs(y_true - y_pred))
#定义该比赛评价模型的对数均方根误差
def log_rmse(net, features, labels):
    # 将⼩于 1 的值设成 1，使得取对数时数值更稳定。
    clipped_preds = nd.clip(net(features), 1, float('inf'))
    rmse = nd.sqrt(2 * loss(clipped_preds.log(), labels.log()).mean())
    return rmse.asscalar()
#定义训练函数
# ctx =d2l.try_gpu()
def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)
    # 这⾥使⽤了 Adam 优化算法。
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
#K折交叉验证 被⽤来选择模型设计并调节超参数
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = nd.concat(X_train, X_part, dim=0)
            y_train = nd.concat(y_train, y_part, dim=0)
    return X_train, y_train, X_valid, y_valid
#在 K 折交叉验证中我们训练 K 次并返回训练和验证的平均误差
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            aul.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                        range(1, num_epochs + 1), valid_ls,
                        ['train', 'valid'])
            plt.show()
        print('fold %d, train rmse: %f, valid rmse: %f' % (
                i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k

train_data = pd.read_csv('./data/train_dataset.csv')
test_data = pd.read_csv('./data/test_dataset.csv')
print('查看前 4 个样本的前 4 个特征、后 2 个特征和标签（信用分）')
print('前4个样本的前4个特征、后2个特征和标签',train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

#将所有的训练和测试数据的 79 个特征按样本连结
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
#预处理数据
# 对连续数值的特征做标准化，设该特征在整个数据集上的均值为 µ，标准差为 σ。那么，我们可以将该特征的每个值先减去 µ 再除以 σ 得到标准化后的每个特征值。对于缺失的特征值，我们将其替换成该特征的均值。
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
all_features = all_features.fillna(all_features.mean())
#将离散数值转成指⽰特征
# 举个例⼦，假设特征 MSZoning ⾥⾯有两个不同的离散值 RL 和 RM，那么这⼀步转换将去掉 MSZoning 特征，并新加两个特征 MSZoning_RL 和 MSZoning_RM，其值为 0 或 1。
all_features = pd.get_dummies(all_features, dummy_na=True)
#通过 values 属性得到 NumPy 格式的数据，并转成 NDArray ⽅便后⾯的训练
n_train = train_data.shape[0]
train_features = nd.array(all_features[:n_train].values)
test_features = nd.array(all_features[n_train:].values)
train_labels = nd.array(train_data.信用分.values).reshape((-1, 1))

#模型选择
# 使⽤⼀组未经调优的超参数并计算交叉验证误差。你可以改动这些超参数来尽可能减小平均测试误差
k, num_epochs, lr, weight_decay, batch_size = 5, 200, 0.003, 0.01 , 512
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
print('%d-fold validation: avg train rmse: %f, avg valid rmse: %f'
        % (k, train_l, valid_l))
    
#定义预测函数  使⽤完整的训练数据集来重新训练模型，并将预测结果存成提交所需要的格式
def train_and_pred(train_features, test_feature, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
    num_epochs, lr, weight_decay, batch_size)
    aul.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).asnumpy()
    test_data['score'] = round(pd.Series(preds.reshape(1, -1)[0]))
    submission = pd.concat([test_data['用户编码'], test_data['score']], axis=1)
    submission.to_csv('submission_scorev1.csv', index=False)

#对测试数据集上的房屋样本做价格预测
train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
