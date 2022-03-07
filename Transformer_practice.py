from cmath import sqrt
from ctypes.wintypes import SERVICE_STATUS_HANDLE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  #函数需要写在forward中，以权重和偏置为输入
import math, copy, time
import torch.autograd import Variable #封装Tensor，用于实现反向传播
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")

class EncoderDecoder(nn.Module): #nn.Module中封装着Pytorch中所有的模块的基类
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__() #继承 nn.Module 的模块在实现自己的 __init__ 函数时，一定要先调用 super().__init__()。
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed  #中文的Embedding
        self.tgt_embed = tgt_embed  #英文的Embedding
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):   #为什么两个都要mask？
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src),src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)    #decoder输入的中文为mask后的中文向量？

class Generator(nn.Module): #Generator为decoder最后的线性层+softmax
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)  #nn.Linear(in_features = , out_features = )
    
    def forward(self, x):  #nn.Xxx与nn.functional.xxx的不同之处在于，前者是类，需要先实例化后才能使用；后者为函数可以直接调用。
        return F.log_softmax(self.proj(x), dim=1) ## 0是对列做归一化，1是对行做归一化; 在softmax后多做一层log运算，防止溢出

def clones(module, N): #MoudleList它是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器。
    # layer -> layer1, layer2 ... layerN
    return nn.MoudleList([copy.deepcopy(module) for _ in range (N)]) #为了建立多个一致的层

class Encoder(nn.Moudle): # N = 6
    # x + mask -> layer1 -> x + mask ->layer2 ... -> layerN -> x ->LayerNorm     这里layer1之后的x都是前一层的输出
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)  #LayerNorm的输入应该是特征的维度

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)   #每一层的输出都是下一层的输入
        return self.norm(x)

class LayerNorm(nn.Moudle):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(feature)) #nn.Parameter的目的是将本身非网络参数的矩阵注册，使其可以被优化
        self.b_2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Moudle):
    # LayerNorm + sublayer(Self-Attenion/Dense) + dropout + 残差连接
	# 为了简单，把LayerNorm放到了前面，这和原始论文稍有不同，原始论文LayerNorm在最后。
    # x -> LayerNorm -> lambda x: self.self_attn(x, x, x, mask) -> dropout -> 
    # 如果神经元被dropout，则输入即为输出
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)  #nn.Dropout(p = 0.3) # 表示每个神经元有0.3的可能性不被激活

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Moudle):
    # x -> LayerNorm -> lambda x: self.self_attn(x, x, x, mask) -> dropout -> x -> LayerNorm -> self.feed_forward -> dropout
    # 根据调用观察可知，SublayerConnection中的sublayer[0]即为lambda x: self.self_attn(x, x, x, mask)，是一个参数
    # 根据调用观察可知，SublayerConnection中的sublayer[1]即为 self.feed_forward
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.moudle):
    # 注意Decoder是由N个DecoderLayer堆叠而成的，使用callable可以直接调用Decoder.forward
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Moudle):
    # 首先注意，不同于EncoderLayer中的2个sublayer，DecoderLayer中有3个sublayer
    # x -> LayerNorm -> lambda x:self.self_attn(x, x, x, tgt_mask) -> dropout -> x -> LayerNorm -> lambda x:self.src_attn(x, x, m, src_mask) -> dropout -> x ->LayerNorm -> self.feed_ward -> dropout
    def __init__ (self, size, self_attn, src_attn, feed_ward, dropout):
        super(DecoderLayer).__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feedward = feed_ward
        self.size = size
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x:self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x:self.src_attn(x, x, m, src_mask))
        return self.sublayer[2](x, self.feed_ward)

# subsequent_mask是Decoder中的Mask机制，为了防止预测信息泄漏
def subsequent_mask(size): 
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('unit8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask = None, dropout = None):  # p_attn即为权重矩阵，在transformer中有两种，一种是非mask，一种为mask
    d_k = query.size(-1)
    scores = torch.matual(query, key.transpose(-2, -1)) \ 
             / math.sqrt(d_k)  # transpose(-2, -1)意味着矩阵乘法只关注后两维，即序列长度，和每一时刻的特征数
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9) #利用一个非常小的数值代替负无穷之后，softmax就会将其输出为0
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matual(p_attn, value), p_attn







