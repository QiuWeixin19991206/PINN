#Wei-Xin Qiu
import tf_last
import tensorflow.compat.v1 as tf#自己加的
tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()#加

# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io  # python读取.mat数据之scipy.io&h5py
from scipy.interpolate import griddata
from pyDOE import lhs  # 拉丁超立方采样
# from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import time
from matplotlib import pyplot, cm
import matplotlib.gridspec as gridspec  # 散点图
from mpl_toolkits.axes_grid1 import make_axes_locatable
# pyplot.rcParams['font.sans-serif'] = ['Times New Roman']
# pyplot.rcParams['axes.unicode_minus']=False
from datetime import datetime

np.random.seed(1234)  # 设置随机数池
tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, data, x0, u1, u2, v1, v2, u1_end, u2_end, v1_end, v2_end, u1_ub, u1_lb, u2_ub, u2_lb, v1_ub, v1_lb, v2_ub, v2_lb, tb, X_f, layers, lb, ub):  # 不需要v0,lb,ub

        X0 = np.concatenate((x0, 0*x0 + lb[1]), 1) # (x0, 0)
        Xend = np.concatenate((x0, 0 * x0 + ub[1]), 1)  # (x0, 0)
        X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
        X_ub = np.concatenate((0*tb + ub[0], tb), 1)  # (ub[0], tb)

        self.data = data

        self.lb = lb
        self.ub = ub

        """MSE0的数据"""
        self.x0 = X0[:, 0:1]  # 取第0列，x,X0是一个两列的数组 ic
        self.t0 = X0[:, 1:2]  # 1
        self.xend = Xend[:, 0:1]  # 取第0列，x,X0是一个两列的数组 ic
        self.tend = Xend[:, 1:2]  # 1
        """MSEB的数据-5"""
        self.x_lb = X_lb[:, 0:1]  # 取-5 bc
        self.t_lb = X_lb[:, 1:2]  # 取tb
        """MSEB的数据5"""
        self.x_ub = X_ub[:, 0:1]  # 取5 bc
        self.t_ub = X_ub[:, 1:2]  # 取tb

        """MSEF的数据"""
        self.x_f = X_f[:, 0:1]  # xf
        self.t_f = X_f[:, 1:2]  # tf

        """实部与虚部 耦合分量1"""
        self.u1 = u1
        self.u1_lb = u1_lb
        self.u1_ub = u1_ub

        self.v1 = v1
        self.v1_lb = v1_lb
        self.v1_ub = v1_ub

        """实部与虚部 耦合分量2"""
        self.u2 = u2
        self.u2_lb = u2_lb
        self.u2_ub = u2_ub

        self.v2 = v2
        self.v2_lb = v2_lb
        self.v2_ub = v2_ub
        """补边界"""
        self.u1_end = u1_end
        self.v1_end = v1_end
        self.u2_end = u2_end
        self.v2_end = v2_end

        """调用函数，获取权重与偏置"""
        # Initialize NNs
        self.layers = layers

        self.weights, self.biases, self.ahs = self.initialize_NN(layers)


        # tf Placeholders      实际值，定义数据的格式 在 TensorFlow 中，
        # 是一个占位符节点，用于在 TensorFlow 图中提供数据输入。它不会进行任何计算，只是在图中标记出输入数据的位置，并且可以在图执行时使用 参数来填充实际的数据，从而实现数据的输入
        self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])  # 输出矩阵的列数,先确定是几个特征向量
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])

        self.xend_tf = tf.placeholder(tf.float32, shape=[None, self.xend.shape[1]])  # 输出矩阵的列数,先确定是几个特征向量
        self.tend_tf = tf.placeholder(tf.float32, shape=[None, self.tend.shape[1]])

        self.u1_tf = tf.placeholder(tf.float32, shape=[None, self.u1.shape[1]])
        self.u1_ub_tf = tf.placeholder(tf.float32, shape=[None, self.u1_ub.shape[1]])
        self.u1_lb_tf = tf.placeholder(tf.float32, shape=[None, self.u1_lb.shape[1]])

        self.u2_tf = tf.placeholder(tf.float32, shape=[None, self.u2.shape[1]])
        self.u2_ub_tf = tf.placeholder(tf.float32, shape=[None, self.u2_ub.shape[1]])
        self.u2_lb_tf = tf.placeholder(tf.float32, shape=[None, self.u2_lb.shape[1]])

        self.v1_tf = tf.placeholder(tf.float32, shape=[None, self.v1.shape[1]])
        self.v1_ub_tf = tf.placeholder(tf.float32, shape=[None, self.v1_ub.shape[1]])
        self.v1_lb_tf = tf.placeholder(tf.float32, shape=[None, self.v1_lb.shape[1]])

        self.v2_tf = tf.placeholder(tf.float32, shape=[None, self.v2.shape[1]])
        self.v2_ub_tf = tf.placeholder(tf.float32, shape=[None, self.v2_ub.shape[1]])
        self.v2_lb_tf = tf.placeholder(tf.float32, shape=[None, self.v2_lb.shape[1]])

        self.u1_end_tf = tf.placeholder(tf.float32, shape=[None, self.u1_end.shape[1]])
        self.u2_end_tf = tf.placeholder(tf.float32, shape=[None, self.u2_end.shape[1]])
        self.v1_end_tf = tf.placeholder(tf.float32, shape=[None, self.v1_end.shape[1]])
        self.v2_end_tf = tf.placeholder(tf.float32, shape=[None, self.v2_end.shape[1]])

        self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, self.x_lb.shape[1]])
        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, self.t_lb.shape[1]])

        self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, self.x_ub.shape[1]])
        self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, self.t_ub.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        # tf Graphs__tf库的计算图
        """计算的是第一个方程"""
        self.u1_pred, self.v1_pred, self.u2_pred, self.v2_pred,_,_,_,_ = self.net_uv1(self.x0_tf, self.t0_tf)#uv网络
        self.u1_end_pred, self.v1_end_pred,self.u2_end_pred, self.v2_end_pred,_,_,_,_  = self.net_uv1(self.xend_tf, self.tend_tf)

        """计算第二个方程的前半部分"""
        self.u1_lb_pred, self.v1_lb_pred, self.u2_lb_pred, self.v2_lb_pred,_,_,_,_  = self.net_uv1(self.x_lb_tf,self.t_lb_tf)

        """计算第二个方程的后半部分"""
        self.u1_ub_pred, self.v1_ub_pred, self.u2_ub_pred, self.v2_ub_pred,_,_,_,_  = self.net_uv1(self.x_ub_tf,self.t_ub_tf)

        """计算第三个方程"""
        self.f_u1_pred, self.f_v1_pred, self.f_u2_pred, self.f_v2_pred = self.net_f_uv(self.x_f_tf, self.t_f_tf)#net_f_uv预测

        self.loss_IC = 50 * (tf.reduce_mean(tf.square(self.u1_tf - self.u1_pred)) +
                        tf.reduce_mean(tf.square(self.v1_tf - self.v1_pred)) +
                        tf.reduce_mean(tf.square(self.u2_tf - self.u2_pred)) +
                        tf.reduce_mean(tf.square(self.v2_tf - self.v2_pred)))


        self.loss_BC = 50 * (tf.reduce_mean(tf.square(self.u2_lb_tf - self.u2_lb_pred)) +
                        tf.reduce_mean(tf.square(self.u2_ub_tf - self.u2_ub_pred)) +
                        tf.reduce_mean(tf.square(self.v2_lb_tf - self.v2_lb_pred)) +
                        tf.reduce_mean(tf.square(self.v2_ub_tf - self.v2_ub_pred)) +
                        tf.reduce_mean(tf.square(self.u1_lb_tf - self.u1_lb_pred)) +
                        tf.reduce_mean(tf.square(self.u1_ub_tf - self.u1_ub_pred)) +
                        tf.reduce_mean(tf.square(self.v1_lb_tf - self.v1_lb_pred)) +
                        tf.reduce_mean(tf.square(self.v1_ub_tf - self.v1_ub_pred)))


        self.loss_res = 0.5*(tf.reduce_mean(tf.square(self.f_u2_pred)) + tf.reduce_mean(tf.square(self.f_v2_pred)) + \
                        tf.reduce_mean(tf.square(self.f_u1_pred)) + tf.reduce_mean(tf.square(self.f_v1_pred)))

        self.loss = self.loss_res + self.loss_BC + self.loss_IC
        # self.loss = tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
        #             tf.reduce_mean(tf.square(self.v0_tf - self.v0_pred)) + \
        #             tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred)) + \
        #             tf.reduce_mean(tf.square(self.v_lb_pred - self.v_ub_pred)) + \
        #             tf.reduce_mean(tf.square(self.f_u_pred)) + \
        #             tf.reduce_mean(tf.square(self.f_v_pred)) + \
        #             tf.reduce_mean(tf.square(self.u1_tf - self.u1_pred)) + \
        #             tf.reduce_mean(tf.square(self.v1_tf - self.v1_pred)) + \
        #             tf.reduce_mean(tf.square(self.u1_lb_pred - self.u1_ub_pred)) + \
        #             tf.reduce_mean(tf.square(self.v1_lb_pred - self.v1_ub_pred)) + \
        #             tf.reduce_mean(tf.square(self.u_x_ub_pred - self.u_x_lb_pred)) + \
        #             tf.reduce_mean(tf.square(self.v_x_ub_pred - self.v_x_lb_pred)) + \
        #             tf.reduce_mean(tf.square(self.u1_x_ub_pred - self.u1_x_lb_pred)) + \
        #             tf.reduce_mean(tf.square(self.v1_x_ub_pred - self.v1_x_lb_pred)) + \
        #             tf.reduce_mean(tf.square(self.f_u1_pred)) + \
        #             tf.reduce_mean(tf.square(self.f_v1_pred))
        # 定义优化器
        # learning_rate = 0.01
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        # self.train_op_SGD = self.optimizer.minimize(self.loss)

        self.optimizer_Adam = tf.train.AdamOptimizer()  # .minimize(self.loss)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)



        # self.global_step = tf.Variable(0, trainable=False)
        # starter_learning_rate = 1e-3    #初始学习率
        # self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step, 1000, 0.9,
        #                                                 staircase=False)                     #每经过200轮次训练后，学习速率变为原来的0.9
        # # Passing global_step to minimize() will increment it at each step.
        # self.train_op_Adam = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
        #                                                                  global_step=self.global_step)

        self.optimizer = tf_last.ScipyOptimizerInterface(self.loss,
                                                                method = 'L-BFGS-B',
                                                                options = {'maxiter': 8000,
                                                                           'maxfun': 8000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})

        """生成一个会话"""
        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=False))

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def swish(self, x):

        return x * tf.sigmoid(x)

    """网络初始化，权重是随机生成的正态分布"""

    def initialize_NN(self, layers):
        weights = []
        biases = []
        ahs = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):  # 权重等于层数减1
            W = self.xavier_init(size=[layers[l], layers[l + 1]])  # 因为layers是一个数组
            a = tf.Variable(1, dtype=tf.float32)
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)  # 偏置数与后一层神经元有关
            weights.append(W)
            biases.append(b)
            ahs.append(a)
        return weights, biases, ahs

    """如何选取x的数据，生成权重"""

    def xavier_init(self, size):
        in_dim = size[0]  # 前一层为输入的维度
        out_dim = size[1]  # 后一层为输出的维度
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))  # 定义了标准差
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev),
                           dtype=tf.float32)  # 指定平均值与标准差的正态分布

    """前向传播的过程"""

    def neural_net_tanh(self, X, weights, biases, ahs):
        num_layers = len(weights) + 1  # 权重的维度就是网络的层数

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0  # ??????

        for l in range(0, num_layers - 2):  # 从数组中取出每层的权重与偏置
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))  # 会计算出n-1个H

        """最后一层输出"""
        W = weights[-1]  # 最后一层权重
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)  # 最后一层是线性的
        return Y  # 神经网络的输出

    def neural_net_sin(self, X, weights, biases, ahs):
        num_layers = len(weights) + 1  # 权重的维度就是网络的层数

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0  # ??????

        for l in range(0, num_layers - 2):  # 从数组中取出每层的权重与偏置
            W = weights[l]
            b = biases[l]
            H = tf.sin(20*ahs[l] * tf.add(tf.matmul(H, W), b))  # 会计算出n-1个H

        """最后一层输出"""
        W = weights[-1]  # 最后一层权重
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)  # 最后一层是线性的
        return Y  # 神经网络的输出

    def neural_net_swish(self, X, weights, biases, ahs):
        num_layers = len(weights) + 1  # 权重的维度就是网络的层数

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0  # ??????

        for l in range(0, num_layers - 2):  # 从数组中取出每层的权重与偏置
            W = weights[l]
            b = biases[l]
            H = tf.nn.softplus(20 * ahs[l] * tf.add(tf.matmul(H, W), b))  # 会计算出n-1个H

        """最后一层输出"""
        W = weights[-1]  # 最后一层权重
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)  # 最后一层是线性的
        return Y

    def neural_net_cos(self, X, weights, biases, ahs):
        num_layers = len(weights) + 1  # 权重的维度就是网络的层数

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0  # ??????

        for l in range(0, num_layers - 2):  # 从数组中取出每层的权重与偏置
            W = weights[l]
            b = biases[l]
            H = tf.cos(20 * ahs[l] * tf.add(tf.matmul(H, W), b))  # 会计算出n-1个H

        """最后一层输出"""
        W = weights[-1]  # 最后一层权重
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)  # 最后一层是线性的
        return Y

    """反向传播过程"""

    def net_uv1(self, x, t):
        X = tf.concat([x, t], 1)  # 将数组x，y进行列拼接X=X0

        u1v1 = self.neural_net_tanh(X, self.weights, self.biases, self.ahs)  # loss函数对应于tanh
        u1 = u1v1[:, 0:1]  # x的意义有点商榷，输出的Y第一列是实部，第二列是虚部；????
        v1 = u1v1[:, 1:2]
        u2 = u1v1[:, 2:3]  # x的意义有点商榷，输出的Y第一列是实部，第二列是虚部；????
        v2 = u1v1[:, 3:4]
        u1_ = u1v1[:, 4:5]  # x的意义有点商榷，输出的Y第一列是实部，第二列是虚部；????
        v1_ = u1v1[:, 5:6]
        u2_ = u1v1[:, 6:7]  # x的意义有点商榷，输出的Y第一列是实部，第二列是虚部；????
        v2_ = u1v1[:, 7:8]

        return u1, v1, u2, v2, u1_, v1_, u2_, v2_

    """惩罚项"""

    def net_f_uv(self, x, t):  # 对对应的位置求导：
        u1, v1, u2, v2 ,real1, image1, real2, image2= self.net_uv1(x, t)

        u2_x = tf.gradients(u2, x)[0]
        u2_xx = tf.gradients(u2_x, x)[0]
        v2_x = tf.gradients(v2, x)[0]
        v2_xx = tf.gradients(v2_x, x)[0]

        u1_x = tf.gradients(u1, x)[0]
        u1_xx = tf.gradients(u1_x, x)[0]
        v1_x = tf.gradients(v1, x)[0]
        v1_xx = tf.gradients(v1_x, x)[0]

        u2_t = tf.gradients(u2, t)[0]
        u1_t = tf.gradients(u1, t)[0]
        v2_t = tf.gradients(v2, t)[0]
        v1_t = tf.gradients(v1, t)[0]

        sigma = 1
        '''设的q=u+i*v q*=real+i*image'''
        # f_u1 = 2 * sigma * real1 * u1 ** 2 - 4 * sigma * image1 * u1 * v1 - 2 * sigma * real1 * v1 ** 2 + 2 * sigma * real2 * u2 ** 2 - 4 * sigma * image2 * u2 * v2 - 2 * sigma * real2 * v2 ** 2 - v1_t + u1_xx
        # f_v1 = 2 * sigma * image1 * u1 ** 2 + 4 * sigma * real1 * u1 * v1 - 2 * sigma * image1 * v1 ** 2 + 2 * sigma * image2 * u2 ** 2 + 4 * sigma * real2 * u2 * v2 - 2 * sigma * image2 * v2 ** 2 + u1_t + v1_xx
        # f_u2 = 2 * sigma * real1 * u1 ** 2 - 4 * sigma * image1 * u1 * v1 - 2 * sigma * real1 * v1 ** 2 + 2 * sigma * real2 * u2 ** 2 - 4 * sigma * image2 * u2 * v2 - 2 * sigma * real2 * v2 ** 2 - v2_t + u2_xx
        # f_v2 = 2 * sigma * image1 * u1 ** 2 + 4 * sigma * real1 * u1 * v1 - 2 * sigma * image1 * v1 ** 2 + 2 * sigma * image2 * u2 ** 2 + 4 * sigma * real2 * u2 * v2 - 2 * sigma * image2 * v2 ** 2 + v2_xx + u2_t
        '''设的q=u+i*v q*=real-i*image'''
        '''设的q=u+i*v q*=real-i*image'''
        f_u1 = 2 * sigma * real2 * u2 ** 2 + 4 * sigma * image2 * u2 * v2 - 2 * sigma * real2 * v2 ** 2 + 2 * sigma * real1 * u1 ** 2 + 4 * sigma * image1 * u1 * v1 - 2 * sigma * real1 * v1 ** 2 - v1_t + u1_xx
        f_v1 = -2 * sigma * image2 * u2 ** 2 + 4 * sigma * real2 * u2 * v2 + 2 * sigma * image2 * v2 ** 2 - 2 * sigma * image1 * u1 ** 2 + 4 * sigma * real1 * u1 * v1 + 2 * sigma * image1 * v1 ** 2 + u1_t + v1_xx
        f_u2 = 2 * sigma * real2 * u2 ** 2 + 4 * sigma * image2 * u2 * v2 - 2 * sigma * real2 * v2 ** 2 + 2 * sigma * real1 * u1 ** 2 + 4 * sigma * image1 * u1 * v1 - 2 * sigma * real1 * v1 ** 2 - v2_t + u2_xx
        f_v2 = -2 * sigma * image2 * u2 ** 2 + 4 * sigma * real2 * u2 * v2 + 2 * sigma * image2 * v2 ** 2 - 2 * sigma * image1 * u1 ** 2 + 4 * sigma * real1 * u1 * v1 + 2 * sigma * image1 * v1 ** 2 + u2_t + v2_xx

        return f_u1, f_v1, f_u2, f_v2

    # def callback(self, loss, loss1):#回调，将参数传递到其他代码，且可以执行该段代码
    # print('Loss:', loss, loss1)

    def callback(self, loss):  # 回调，将参数传递到其他代码，且可以执行该段代码
        print('Loss:', loss)

    def train(self, nIter):
        # 训练的数据，占位符
        tf_dict = {self.x0_tf: self.x0, self.t0_tf: self.t0,
                   self.xend_tf: self.xend, self.tend_tf: self.tend,
                   self.u2_tf: self.u2, self.v2_tf: self.v2,
                   self.u1_tf: self.u1, self.v1_tf: self.v1,
                   self.u1_end_tf: self.u1_end, self.v1_end_tf: self.v1_end,
                   self.u2_end_tf: self.u2_end, self.v2_end_tf: self.v2_end,
                   self.u2_lb_tf: self.u2_lb, self.v2_lb_tf: self.v2_lb,
                   self.u2_ub_tf: self.u2_ub, self.v2_ub_tf: self.v2_ub,
                   self.u1_lb_tf: self.u1_lb, self.v1_lb_tf: self.v1_lb,
                   self.u1_ub_tf: self.u1_ub, self.v1_ub_tf: self.v1_ub,
                   self.x_lb_tf: self.x_lb, self.t_lb_tf: self.t_lb,
                   self.x_ub_tf: self.x_ub, self.t_ub_tf: self.t_ub,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}
        MSE_history = []#存loss的记录
        tiem = []#存时间，每训练 次的时间，用于打印

        first_time = time.time()
        start_time = first_time
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)  # 调用优化器，运行反向传播；nIter循环的次数  train_op  train_op_v
            # self.sess.run(self.train_op_Adam1, tf_dict)
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time  # 训练10次花费的时间
                loss_value = self.sess.run(self.loss, tf_dict)
                loss_value1 = self.sess.run(self.loss_IC, tf_dict)
                loss_value2 = self.sess.run(self.loss_res, tf_dict)
                loss_value3 = self.sess.run(self.loss_BC, tf_dict)
                Total = datetime.fromtimestamp(time.time() - first_time).strftime("%M:%S")
                # print('It: %d, Loss: %.3e, IC: %.3e, BC: %.3e, res: %.3e,Time: %.2f' %
                #       (it, loss_value, loss_value1, loss_value3, loss_value2, elapsed))

                print(f"epoch = {it} " +  # 训练次数
                      f"elapsed = {Total} " +  # 总时间
                      f"(+{elapsed:.2f}) " +  # 每次时间
                      f"loss: = {loss_value:.4e} "  # 损失函数loss
                      f"IC: = {loss_value1:.4e} "  # 损失函数loss
                      f"BC: = {loss_value3:.4e} "  # 损失函数loss
                      f"res: = {loss_value2:.4e} "  # 损失函数loss
                      )

                start_time = time.time()

                if it % 1000 == 0:
                    u1_pred, v1_pred, u2_pred, v2_pred, f_u1_pred, f_v1_pred, f_u2_pred, f_v2_pred = model.predict(X_star)

                    h2_pred = np.sqrt(u2_pred ** 2 + v2_pred ** 2)
                    h1_pred = np.sqrt(u1_pred ** 2 + v1_pred ** 2)
                    # 画散点图
                    H2_pred = griddata(X_star, h2_pred.flatten(), (X, T), method='cubic')
                    H1_pred = griddata(X_star, h1_pred.flatten(), (X, T), method='cubic')

                    data = self.data
                    # 精确解的数据集
                    t = data['tt'].flatten()[:, None]
                    x = data['z'].flatten()[:, None]
                    Exact1 = data['uu1']
                    Exact2 = data['uu2']
                    Exact_u2 = np.real(Exact2)
                    Exact_v2 = np.imag(Exact2)
                    Exact_h2 = np.sqrt(Exact_u2 ** 2 + Exact_v2 ** 2)

                    Exact_u1 = np.real(Exact1)
                    Exact_v1 = np.imag(Exact1)
                    Exact_h1 = np.sqrt(Exact_u1 ** 2 + Exact_v1 ** 2)

                    fig = plt.figure(dpi=130)
                    ax = plt.subplot(3, 1, 1)
                    plt.plot(x, Exact_h1[:, 0], 'b-', linewidth=2, label='Exact')
                    plt.plot(x, H1_pred[0, :], 'r--', linewidth=2, label='Prediction')
                    ax.set_ylabel('$|U1(t,x)|$')
                    ax.set_xlabel('$x$')
                    ax.set_title('$t = %.2f$' % (t[0]), fontsize=10)
                    plt.legend()

                    ax1 = plt.subplot(3, 1, 2)
                    plt.plot(x, Exact_h1[:, 100], 'b-', linewidth=2, label='Exact')
                    plt.plot(x, H1_pred[100, :], 'r--', linewidth=2, label='Prediction')
                    ax1.set_ylabel('$|U1(t,x)|$')
                    ax1.set_xlabel('$x$')
                    ax1.set_title('$t = %.2f$' % (t[100]), fontsize=10)
                    plt.legend()

                    ax2 = plt.subplot(3, 1, 3)
                    plt.plot(x, Exact_h1[:, 150], 'b-', linewidth=2, label='Exact')
                    plt.plot(x, H1_pred[150, :], 'r--', linewidth=2, label='Prediction')
                    ax2.set_ylabel('$|U1(t,x)|$')
                    ax2.set_xlabel('$x$')
                    ax2.set_title('$t = %.2f$' % (t[150]), fontsize=10)
                    plt.legend()
                    fig.tight_layout()
                    plt.savefig('./photo/Q1(t,x)_{}.png'.format(it/1000))

                    fig = plt.figure(dpi=130)
                    ax = plt.subplot(3, 1, 1)
                    plt.plot(x, Exact_h2[:, 0], 'b-', linewidth=2, label='Exact')
                    plt.plot(x, H2_pred[0, :], 'r--', linewidth=2, label='Prediction')
                    ax.set_ylabel('$|U2(t,x)|$')
                    ax.set_xlabel('$x$')
                    ax.set_title('$t = %.2f$' % (t[0]), fontsize=10)
                    plt.legend()

                    ax1 = plt.subplot(3, 1, 2)
                    plt.plot(x, Exact_h2[:, 100], 'b-', linewidth=2, label='Exact')
                    plt.plot(x, H2_pred[100, :], 'r--', linewidth=2, label='Prediction')
                    ax1.set_ylabel('$|U2(t,x)|$')
                    ax1.set_xlabel('$x$')
                    ax1.set_title('$t = %.2f$' % (t[100]), fontsize=10)
                    plt.legend()

                    ax2 = plt.subplot(3, 1, 3)
                    plt.plot(x, Exact_h2[:, 150], 'b-', linewidth=2, label='Exact')
                    plt.plot(x, H2_pred[150, :], 'r--', linewidth=2, label='Prediction')
                    ax2.set_ylabel('$|U2(t,x)|$')
                    ax2.set_xlabel('$x$')
                    ax2.set_title('$t = %.2f$' % (t[150]), fontsize=10)
                    plt.legend()
                    fig.tight_layout()
                    plt.savefig('./photo/Q2(t,x)_{}.png'.format(it/1000))

                    fig = plt.figure("预测演化图1", dpi=130, facecolor=None, edgecolor=None, frameon=None)
                    ax = fig.add_subplot(projection='3d')
                    surf = ax.plot_surface(X, T, H1_pred, cmap='coolwarm', rstride=1, cstride=1,
                                           linewidth=0, antialiased=False)
                    # ax.grid(False)
                    ax.set_xlabel('$x$')
                    ax.set_ylabel('$t$')
                    ax.set_zlabel('$|U1(t,x)|$')
                    plt.savefig('./photo/预测演化图1_{}.png'.format(it/1000));

                    fig = plt.figure("预测演化图2", dpi=130, facecolor=None, edgecolor=None, frameon=None)
                    ax = fig.add_subplot(projection='3d')
                    surf = ax.plot_surface(X, T, H2_pred, cmap='coolwarm', rstride=1, cstride=1,
                                           linewidth=0, antialiased=False)
                    # ax.grid(False)
                    ax.set_xlabel('$x$')
                    ax.set_ylabel('$t$')
                    ax.set_zlabel('$|U2(t,x)|$')
                    plt.savefig('./photo/预测演化图2_{}.png'.format(it/1000));

                    plt.close("all")

                MSE_history.append(loss_value)
                tiem.append(elapsed)
                # MSE_history1.append(loss1_value)
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss],
                                loss_callback = self.callback)
        return MSE_history, tiem
        # MSE_history1

    def predict(self, X_star):

        tf_dict = {self.x0_tf: X_star[:, 0:1], self.t0_tf: X_star[:, 1:2]}

        u2_star = self.sess.run(self.u2_pred, tf_dict)
        v2_star = self.sess.run(self.v2_pred, tf_dict)

        u1_star = self.sess.run(self.u1_pred, tf_dict)
        v1_star = self.sess.run(self.v1_pred, tf_dict)

        tf_dict = {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]}

        f_u2_star = self.sess.run(self.f_u2_pred, tf_dict)
        f_v2_star = self.sess.run(self.f_v2_pred, tf_dict)
        f_u1_star = self.sess.run(self.f_u1_pred, tf_dict)
        f_v1_star = self.sess.run(self.f_v1_pred, tf_dict)


        return u1_star, v1_star, u2_star, v2_star, f_u1_star, f_v1_star,f_u2_star, f_v2_star
        # return u_star, v_star, u1_star, v1_star, f_u_star, f_v_star


if __name__ == "__main__":
    noise = 0.0

    # Doman bounds
    lb = np.array([-40.0, 0.0])  # 左边界
    ub = np.array([39.6875, 2.0])  # 右边界
    N0 = 100  # 在h(0,x)解析的数据上选取50个数据点50
    N_b = 100  # 随机采样50个数据点，用于周期性边界条件
    N_f = 10000  # 用于方程5
    layers = [2, 100, 100, 100, 100, 100, 100, 100, 8]  # 6层；
    data = scipy.io.loadmat(r'C:\Users\hcxy\Desktop\工作2\114汇总\非简并单孤子c\单网络+prior\c.mat')
    # 精确解的数据集
    t = data['tt'].flatten()[:, None]
    x = data['z'].flatten()[:, None]
    Exact1 = data['uu1']
    Exact2 = data['uu2']
    Exact_u2 = np.real(Exact2)
    Exact_v2 = np.imag(Exact2)
    Exact_h2 = np.sqrt(Exact_u2**2 + Exact_v2**2)

    Exact_u1 = np.real(Exact1)
    Exact_v1 = np.imag(Exact1)
    Exact_h1 = np.sqrt(Exact_u1 ** 2 + Exact_v1 ** 2)

    X, T = np.meshgrid(x, t)  # 以x,y生成坐标

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))  # 将x,t都变成一维的，即行变为一，在进行拼接。取20000个2维的数据。
    u2_star = Exact_u2.T.flatten()[:,None]#将网络输出的实部结果变为一维，
    v2_star = Exact_v2.T.flatten()[:,None]#将网络输出的虚部结果变为一维
    h2_star = Exact_h2.T.flatten()[:,None]#变为一维的

    u1_star = Exact_u1.T.flatten()[:, None]  # 将网络输出的实部结果变为一维，
    v1_star = Exact_v1.T.flatten()[:, None]  # 将网络输出的虚部结果变为一维
    h1_star = Exact_h1.T.flatten()[:, None]  # 变为一维的

    ###########################

    idx_x = np.random.choice(x.shape[0], N0, replace=False)#输出数组x的行数输出为(0——256-1)，无放回抽样，抽取50个行的位置
    x0 = x[idx_x,:]#取对应列的数据
    #精确解，第idx_x行，第一列的数据 256*201数据集
    u1 = Exact_u1[idx_x,0:1]#因为t=t0，所以去取第一列
    u2 = Exact_u2[idx_x,0:1]#因为t=t0，所以去取第一列
    v1 = Exact_v1[idx_x,0:1]#因为t=t0，所以去取第一列
    v2 = Exact_v2[idx_x,0:1]#因为t=t0，所以去取第一列
    u1_end = Exact_u1[idx_x,200:201]#因为t=t0，所以去取第一列
    u2_end = Exact_u2[idx_x,200:201]#因为t=t0，所以去取第一列
    v1_end = Exact_v1[idx_x,200:201]#因为t=t0，所以去取第一列
    v2_end = Exact_v2[idx_x,200:201]#因为t=t0，所以去取第一列


    idx_t = np.random.choice(t.shape[0], N_b, replace=False)  # 从t中随机抽样50个数据
    tb = t[idx_t, :]
    u1_lb = Exact_u1[0:1, idx_t]  # 因为t=t0，所以去取第一列
    u1_ub = Exact_u1[255:256, idx_t]
    u2_lb = Exact_u2[0:1, idx_t]  # 因为t=t0，所以去取第一列
    u2_ub = Exact_u2[255:256, idx_t]

    v1_lb = Exact_v1[0:1, idx_t]  # 因为t=t0，所以去取第一列
    v1_ub = Exact_v1[255:256, idx_t]
    v2_lb = Exact_v2[0:1, idx_t]  # 因为t=t0，所以去取第一列
    v2_ub = Exact_v2[255:256, idx_t]

    X_f = lb + (ub - lb) * lhs(2, N_f)  # 拉丁超立方采样x_f是一个N_f*2维的数据,???

    model = PhysicsInformedNN(data, x0, u1, u2, v1, v2, u1_end, u2_end, v1_end, v2_end, u1_ub, u1_lb, u2_ub, u2_lb, v1_ub, v1_lb, v2_ub, v2_lb, tb, X_f, layers, lb, ub)

    start_time = time.time()
    It=10000
    MSE_hist, tiem = model.train(It)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))#结束后打印时间
    u1_pred, v1_pred, u2_pred, v2_pred, f_u1_pred, f_v1_pred,  f_u2_pred, f_v2_pred = model.predict(X_star)

    h2_pred = np.sqrt(u2_pred ** 2 + v2_pred ** 2)
    h1_pred = np.sqrt(u1_pred ** 2 + v1_pred ** 2)

    error_u2 = np.linalg.norm(u2_star - u2_pred, 2) / np.linalg.norm(u2_star, 2)
    error_v2 = np.linalg.norm(v2_star - v2_pred, 2) / np.linalg.norm(v2_star, 2)
    error_h2 = np.linalg.norm(h2_star - h2_pred, 2) / np.linalg.norm(h2_star, 2)
    print('Error u2: %e' % (error_u2))
    print('Error v2: %e' % (error_v2))
    print('Error h2: %e' % (error_h2))

    error_u1 = np.linalg.norm(u1_star - u1_pred, 2) / np.linalg.norm(u1_star, 2)
    error_v1 = np.linalg.norm(v1_star - v1_pred, 2) / np.linalg.norm(v1_star, 2)
    error_h1 = np.linalg.norm(h1_star - h1_pred, 2) / np.linalg.norm(h1_star, 2)
    print('Error u1: %e' % (error_u1))
    print('Error v1: %e' % (error_v1))
    print('Error h1: %e' % (error_h1))

    # 画散点图
    U2_pred = griddata(X_star, u2_pred.flatten(), (X, T), method='cubic')
    V2_pred = griddata(X_star, v2_pred.flatten(), (X, T), method='cubic')
    H2_pred = griddata(X_star, h2_pred.flatten(), (X, T), method='cubic')

    FU2_pred = griddata(X_star, f_u2_pred.flatten(), (X, T), method='cubic')
    FV2_pred = griddata(X_star, f_v2_pred.flatten(), (X, T), method='cubic')

    U1_pred = griddata(X_star, u1_pred.flatten(), (X, T), method='cubic')
    V1_pred = griddata(X_star, v1_pred.flatten(), (X, T), method='cubic')
    H1_pred = griddata(X_star, h1_pred.flatten(), (X, T), method='cubic')

    FU1_pred = griddata(X_star, f_u1_pred.flatten(), (X, T), method='cubic')
    FV1_pred = griddata(X_star, f_v1_pred.flatten(), (X, T), method='cubic')

    ######################################################################
    ############################# Plotting ##########################                       #####
    ######################################################################
    X0 = np.concatenate((x0, 0 * x0 + lb[1]), 1)  # (x0, 0)
    Xend = np.concatenate((x0, 0 * x0 + ub[1]), 1)  # (x0, 0)
    X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)(-5,tb)空间确定，时间不定
    X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)(5,tb)空间确定，时间不定
    X_u_train = np.vstack([X0, X_lb, X_ub])  # 拼接成一列的数组

    # 图形布局
    ####### Row 1: h(t,x) slices ##################
    import pandas as pd
    test = pd.DataFrame(columns=['1'], data=MSE_hist)
    # 2.数据保存，index表示是否显示行名，sep数据分开符
    test.to_csv('loss.csv', index=False, sep=',')

    fig = plt.figure("loss", dpi=100, facecolor=None, edgecolor=None, frameon=True)
    plt.plot(range(1, It + 1, 10), MSE_hist, 'r-', linewidth=1, label='learning rate=0.0001')  # + 1前面的数就是迭代次数
    plt.xlabel('$\#$ iterations')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.savefig('loss.png')

    fig = plt.figure(dpi=130)
    ax = plt.subplot(3, 1, 1)
    plt.plot(x, Exact_h1[:, 0], 'b-', linewidth=2, label='Exact')
    plt.plot(x, H1_pred[0, :], 'r--', linewidth=2, label='Prediction')
    ax.set_ylabel('$|U1(t,x)|$')
    ax.set_xlabel('$x$')
    ax.set_title('$t = %.2f$' % (t[0]), fontsize=10)
    plt.legend()

    ax1 = plt.subplot(3, 1, 2)
    plt.plot(x, Exact_h1[:, 100], 'b-', linewidth=2, label='Exact')
    plt.plot(x, H1_pred[100, :], 'r--', linewidth=2, label='Prediction')
    ax1.set_ylabel('$|U1(t,x)|$')
    ax1.set_xlabel('$x$')
    ax1.set_title('$t = %.2f$' % (t[100]), fontsize=10)
    plt.legend()

    ax2 = plt.subplot(3, 1, 3)
    plt.plot(x, Exact_h1[:, 150], 'b-', linewidth=2, label='Exact')
    plt.plot(x, H1_pred[150, :], 'r--', linewidth=2, label='Prediction')
    ax2.set_ylabel('$|U1(t,x)|$')
    ax2.set_xlabel('$x$')
    ax2.set_title('$t = %.2f$' % (t[150]), fontsize=10)
    plt.legend()
    fig.tight_layout()
    plt.savefig('Q1(t,x).png')

    fig = plt.figure(dpi=130)
    ax = plt.subplot(3, 1, 1)
    plt.plot(x, Exact_h2[:, 0], 'b-', linewidth=2, label='Exact')
    plt.plot(x, H2_pred[0, :], 'r--', linewidth=2, label='Prediction')
    ax.set_ylabel('$|U2(t,x)|$')
    ax.set_xlabel('$x$')
    ax.set_title('$t = %.2f$' % (t[0]), fontsize=10)
    plt.legend()

    ax1 = plt.subplot(3, 1, 2)
    plt.plot(x, Exact_h2[:, 100], 'b-', linewidth=2, label='Exact')
    plt.plot(x, H2_pred[100, :], 'r--', linewidth=2, label='Prediction')
    ax1.set_ylabel('$|U2(t,x)|$')
    ax1.set_xlabel('$x$')
    ax1.set_title('$t = %.2f$' % (t[100]), fontsize=10)
    plt.legend()

    ax2 = plt.subplot(3, 1, 3)
    plt.plot(x, Exact_h2[:, 150], 'b-', linewidth=2, label='Exact')
    plt.plot(x, H2_pred[150, :], 'r--', linewidth=2, label='Prediction')
    ax2.set_ylabel('$|U2(t,x)|$')
    ax2.set_xlabel('$x$')
    ax2.set_title('$t = %.2f$' % (t[150]), fontsize=10)
    plt.legend()
    fig.tight_layout()
    plt.savefig('Q2(t,x).png')

    fig = plt.figure("误差图2", dpi=130, facecolor=None, edgecolor=None, frameon=None)
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X, T, Exact_h2.T-H2_pred, rstride=1, cstride=1, cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    # ax.grid(False)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$z$')
    ax.set_zlabel('$|U_2(z,t)|$')
    plt.savefig('误差图2.png')

    fig = plt.figure("误差图1", dpi=130, facecolor=None, edgecolor=None, frameon=None)
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X, T, Exact_h1.T-H1_pred, rstride=1, cstride=1, cmap=cm.viridis,
                           linewidth=0, antialiased=False)
    # ax.grid(False)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$z$')
    ax.set_zlabel('$|U_1(z,t)|$')
    plt.savefig('误差图1.png')

    fig6 = plt.figure('Error Dynamics2', dpi=130)
    ax = fig6.add_subplot(1, 1, 1)
    # ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'rx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=4,
    # clip_on=False)
    h = ax.imshow(Exact_h2 - H2_pred.T, interpolation='nearest', cmap='YlGnBu',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    plt.colorbar(h)
    ax.set_ylabel('$t$')
    ax.set_xlabel('$z$')
    plt.title('Error Dynamics')
    plt.savefig('Error Dynamics2.png')

    fig61 = plt.figure('Error Dynamics1', dpi=130)
    ax = fig61.add_subplot(1, 1, 1)
    # ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'rx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=4,
    # clip_on=False)
    h = ax.imshow(Exact_h1 - H1_pred.T, interpolation='nearest', cmap='YlGnBu',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    plt.colorbar(h)
    ax.set_ylabel('$t$')
    ax.set_xlabel('$z$')
    plt.title('Error Dynamics1')
    plt.savefig('Error Dynamics1.png')

    fig = plt.figure('实际h1(t,x)', dpi=130)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'rx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=4,
            clip_on=False)
    h = ax.imshow(Exact_h1, interpolation='nearest', cmap='YlGnBu',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    plt.colorbar(h)
    ax.set_ylabel('$x$')
    ax.set_xlabel('$t$')
    plt.title('Exact Dynamics1')
    plt.savefig('实际h1(t,x).png')

    fig = plt.figure('实际h2(t,x)', dpi=130)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'rx', label='Data (%d points)' % (X_u_train.shape[0]), markersize=4,
            clip_on=False)
    h = ax.imshow(Exact_h2, interpolation='nearest', cmap='YlGnBu',
                  extent=[lb[1], ub[1], lb[0], ub[0]],
                  origin='lower', aspect='auto')
    plt.colorbar(h)
    ax.set_ylabel('$x$')
    ax.set_xlabel('$t$')
    plt.title('Exact Dynamics2')
    plt.savefig('实际h2(t,x).png')

    fig = plt.figure("实际演化图1", dpi=130, facecolor=None, edgecolor=None, frameon=None)
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X, T, Exact_h1.T, cmap='coolwarm', rstride=1, cstride=1,
                           linewidth=0, antialiased=False)
    # ax.grid(False)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    ax.set_zlabel('$|U1(t,x)|$')
    plt.savefig('实际演化图1.png');

    fig = plt.figure("预测演化图1", dpi=130, facecolor=None, edgecolor=None, frameon=None)
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X, T, H1_pred, cmap='coolwarm', rstride=1, cstride=1,
                           linewidth=0, antialiased=False)
    # ax.grid(False)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    ax.set_zlabel('$|U1(t,x)|$')
    plt.savefig('预测演化图1.png');

    fig = plt.figure("实际演化图2", dpi=130, facecolor=None, edgecolor=None, frameon=None)
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X, T, Exact_h2.T, cmap='coolwarm', rstride=1, cstride=1,
                           linewidth=0, antialiased=False)
    # ax.grid(False)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    ax.set_zlabel('$|U2(t,x)|$')
    plt.savefig('实际演化图2.png');

    fig = plt.figure("预测演化图2", dpi=130, facecolor=None, edgecolor=None, frameon=None)
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X, T, H2_pred, cmap='coolwarm', rstride=1, cstride=1,
                           linewidth=0, antialiased=False)
    # ax.grid(False)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$t$')
    ax.set_zlabel('$|U2(t,x)|$')
    plt.savefig('预测演化图2.png');

    scipy.io.savemat('datetf1.mat',
                     {'x': x, 't': t,
                      'U1_pred': h1_pred, 'U2_pred': h2_pred,
                      'loss': MSE_hist, 'Xstar': X_star,
                      'U1_exact': h1_star, 'U2_exact': h2_star,
                      'error_u1': error_u1, 'error_v1': error_v1, 'error_u2': error_u2, 'error_v2': error_v2, 'error_h1': error_h1, 'error_h2': error_h2,
                      })

    print("over!")
    plt.show()
