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

"""
1. Transformer的结构可以分为Encoder与Decoder；以中译英为例，则Ecoder的输入为中文，Decoder的输出为英文
2. 编码器由N个相同的层stack而成，每一层中包含两个子层
"""

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

def attention(query, key, value, mask = None, dropout = None): 
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # transpose(-2, -1)意味着矩阵乘法只关注后两维，即序列长度，和每一时刻的特征数
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9) #利用一个非常小的数值代替负无穷之后，softmax就会将其输出为0
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matual(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        # h代表头数，d_model代表词嵌入的维度；d_k代表得到每个头获得的分割词向量维度d_k （应该就是将每个单词的维度按照h个数量，切割，每一个头获得的维度为 d_model/h）
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0 #assert 声明，主张;作用：检查程序，不符合条件即终止程序
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, query, key, value, mask = None):
        if mask is not None:
            mask = mask.unsqueeze(1) # 在矩阵的第二维添加一个维度，即若原来为[256,256]，则为[256，1，256]
        nbatches = query.size(0) #代表有多少条样本
        # 首先将Q, K, V三个矩阵与三个线性层用zip组合起来，可以实现并行计算；l(x).view() 意味着要将l(x)的输出转为这个维度，然后再将第一维和第二维调换，即(nbatches, self.h, -1, self.d_k), 目的是让句子长度与分割后的词向量维度相邻
        # 利用卷积的多通道会刚好理解一些，（batchsize, 通道数，句子长度，词向量维度）
        # 实现batch内所有通道的映射 d_model -> d_k * h
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # 对所有通道的线性映射实现Attention，并得到加权矩阵和权重矩阵
        x, self.attn = attention(query, key, value, mask = mask, dropout = self.dropout)
        # 将得到的每个头的4维张量的维度转换为输入时的维度，为了合并；contiguous是为了让转置后的张量可以使用view()函数
        x = x.transpose(1, 2).contiguous()\
            .view(nbatches, -1, self.h * self.d_k)
        # 经过线性层后输出
        return self.linears[-1](x) 

# d_model = 512;d_ff = 2048
class PositionwiseFeedForward(nn.Moudle):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        # 希望输入和输出的维度相同，减小模型的复杂度
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionEmbedding(nn.Module):
    # 在一句话的每一个单词的编码中加入位置信息
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionEmbedding, self).__init__()
        self.dropout = nn.dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2014, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionEmbedding(d_model, dropout)
    # 逻辑关系太明显了，自己看吧
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    # 参数初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

# 到此为止，Transformer的结构就构建完啦，学会了嘛！
# 以下是有关于训练的一些技巧还有模型的设置，我放在第二个文件啦！


