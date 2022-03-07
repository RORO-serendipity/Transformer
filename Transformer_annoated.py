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
1. Transformer的结构可以分为Encoder与Decoder;以中译英为例,则Ecoder的输入为中文,Decoder的输出为英文
2. 编码器由N个相同的EcoderLayer堆叠而成,每一层中包含两个子层:
   a) 第一个子层为Multi-Head Attention (多头注意力机制;多通道注意力机制)+ LayerNorm + 残差连接
   b) 第二个子层为Feed Forward + LayerNorm + 残差连接
3. 解码器同样由N个相同的DcoderLayer堆叠而成, 每一层中包含三个子层:
   a) 第一个子层为一个Masked Multi-Head Attention + LayerNorm + 残差连接 (Masked的作用是防止预测信息泄漏, 即预测第t时刻就不能输入t时刻之后的信息)
   b) 第二个子层为Multi-Head Attention (多头注意力机制;多通道注意力机制) + LayerNorm + 残差连接 (Decoder在这一层的输入来自于两部分, 一部分是来自第一个子层的Query, 一部分是来自Encoder的Key和Value)
   c) 第三个子层为Feed Forward + LayerNorm + 残差连接
4. 两部分模型的输入均由 Input Embedding和 Position Embedding两部分组合而成
5. Transformer的Decoder输出的是概率问题, 本质实现的应该是一种匹配问题。经过模型的训练, 选择字典库中概率最大的词作为本次的输出。
6. 在训练过程中, Transformer的Decoder的输入初始应该是起始符, 然后训练出第一个单词; 由起始符+第一个单词 -> 第二个单词
7. 在上述的训练过程, Decoder的每次输入都是已经确定的矩阵, 即在训练的过程中, 即使预测错误也没关系, Decoder在第二次的输出, 不会作为第三次的输入
(Teacher-Forcing: Teacher Forcing工作原理: 在训练过程的t时刻, 使用训练数据集的期望输出或实际输出: y(t), 作为下一时间步骤的输入: x(t+1)而不是使用模型生成的输出h(t))
"""

# Transformer_practice文件与源码的顺序是相同的，顺序很乱；这个程序按照模型的顺序重新调整了一下。

# Embedding 层

class Embeddings(nn.Moudle):
    def __init__(self, d_model, vocab):
        """
        d_model: 指的是词嵌入的维度;例如,“我”这个词,编码后维度为512
        vocab: 词表的大小
        nn.Embedding: 词典的大小尺寸,比如总共出现5000个词,那就输入5000
        """
        super(Embeddings).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionEmbedding(nn.Module):
    # 在一句话的每一个单词的编码中加入位置信息
    """
    位置编码器类的初始化函数, 一共三个参数分别为:
    d_model: 词嵌入的维度
    dropout: 比例
    max_then: 每个句子最大的长度
    """
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

# 从接下来的代码中可以看出，Transformer的编码器是并行计算的，而解码器是循环计算的

def clones(module, N): 
    #MoudleList它是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器。
    # layer -> layer1, layer2 ... layerN
    return nn.MoudleList([copy.deepcopy(module) for _ in range (N)])

# 由于Multi-Head Attention与LayerNorm 同时存在于编码器和解码器中，因此先构建。
def attention(query, key, value, mask = None, dropout = None): 
    # 取query的最后一维的大小，对应为词嵌入的维度（也就是d_model = 512)
    d_k = query.size(-1)
    # transpose(-2, -1)意味着矩阵乘法只关注后两维，即语句的长度和词嵌入的维度 “我爱你，中国”[30, 8, 6, 512]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) 
    # 判断是否进行掩码张量；masked_fill的使用方法为，mask中取值为0的地方，对应于scores张量用1e-9替换；这里的mask是用于padding_mask
    """
    假如现在有一组batchsize为3, 长度为4的序列
    [['a', 'b', 'c', 'd' ],
     ['a', 'b', <Pad>, <Pad>],
     ['a', 'b', 'c', <Pad>]]
    为了对齐不同的序列, 以最长的序列为基准, 利用padding把序列补齐。
    但是Pad本身也是可以被Embedding的, 而a与<Pad>之间计算是没有任何意义的, 所以要将Pad的地方换掉。换为一个极小的数。
    """
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 将p_attn与value张量相乘获得最终的query注意力表示，同时返回注意力张量
    return torch.matual(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        # h代表头数，d_model代表词嵌入的维度；d_k代表得到每个头获得的分割词向量维度d_k
        super(MultiHeadAttention, self).__init__()
        #assert 声明，主张;作用：检查程序，不符合条件即终止程序
        assert d_model % h == 0 
        # 将每个单词的维度按照h个数量，切割，每一个头获得的维度为 d_model/h）
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, query, key, value, mask = None):
        if mask is not None:
            # 在矩阵的第二维添加一个维度，代表n个头中的一个，所有头的mask都是相同的
            mask = mask.unsqueeze(1) 
        # 代表有多少条样本
        nbatches = query.size(0) 
        # 首先将Q, K, V三个矩阵与三个线性层用zip组合起来，可以实现并行计算；l(x).view() 意味着要将l(x)的输出转为这个维度，然后再将第一维和第二维调换，即(nbatches, self.h, -1, self.d_k), 目的是让句子长度与分割后的词向量维度相邻
        # 利用卷积的多通道会刚好理解一些，（batchsize, 通道数，句子长度，词向量维度）
        # l(x)的维度应该是（d_model, d_model）
        # 实现batch内所有通道的映射 d_model -> d_k * h
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # 对所有通道的线性映射实现Attention，并得到加权矩阵和权重矩阵
        x, self.attn = attention(query, key, value, mask = mask, dropout = self.dropout)
        # 将得到的每个头的4维张量的维度转换为输入时的维度，为了合并；contiguous是为了让转置后的张量可以使用view()函数；
        x = x.transpose(1, 2).contiguous()\
            .view(nbatches, -1, self.h * self.d_k) 
        # 经过线性层后输出
        return self.linears[-1](x) 

# LayerNorm
class LayerNorm(nn.Moudle):
    def __init__(self, features, eps=1e-6):
        """
        feature: 词嵌入的维度
        eps: 一个足够小的数, 防止分母为0
        """
        super(LayerNorm, self).__init__()
        # nn.Parameter的目的是将本身非网络参数的矩阵注册，使其可以被优化; 在这里是作为调节因此，保持样本的特征
        self.a_2 = nn.Parameter(torch.ones(feature)) 
        self.b_2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2

# FeedForward
class PositionwiseFeedForward(nn.Moudle):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        # 希望输入和输出的维度相同，减小模型的复杂度
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.dropout(p=dropout)

    def forward(self, x):
        # x -> 第一个线性层 -> relu -> dropout -> 第二个线性层
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class SublayerConnection(nn.Moudle):
    # LayerNorm + sublayer(Self-Attenion/Dense) + dropout + 残差连接
	# 为了简单，把LayerNorm放到了前面，这和原始论文稍有不同，原始论文LayerNorm在最后。
    # x -> LayerNorm -> lambda x: self.self_attn(x, x, x, mask) -> dropout -> y + x
    # 如果神经元被dropout，则输入即为输出
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)  #nn.Dropout(p = 0.3) # 表示每个神经元有0.3的可能性不被激活

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Moudle):
    # x -> LayerNorm -> lambda x: self.self_attn(x, x, x, mask) -> dropout -> y1 + x -> LayerNorm -> self.feed_forward -> dropout -> y2 + y1 + x
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

class Encoder(nn.Moudle): # N = 6
    # x + mask -> layer1 -> y1 + mask ->layer2 ... -> layerN -> yN ->LayerNorm   
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)  #LayerNorm的输入应该是特征的维度

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)   #每一层的输出都是下一层的输入
        return self.norm(x)

# Decoder中的掩码: subsequent_mask是Decoder中的Mask机制，为了防止预测信息泄漏
def subsequent_mask(size): 
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('unit8')
    return torch.from_numpy(subsequent_mask) == 0

# Generator为decoder最后的线性层+softmax
class Generator(nn.Module): 
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # nn.Linear(in_features = , out_features = )
        self.proj = nn.Linear(d_model, vocab) 
    
    # nn.Xxx与nn.functional.xxx的不同之处在于，前者是类，需要先实例化后才能使用；后者为函数可以直接调用。
    def forward(self, x): 
        # 0是对列做归一化，1是对行做归一化; 在softmax后多做一层log运算，防止溢出
        return F.log_softmax(self.proj(x), dim=1)

# 首先注意，不同于EncoderLayer中的2个sublayer，DecoderLayer中有3个sublayer
# x -> LayerNorm -> lambda x:self.self_attn(x, x, x, tgt_mask) -> dropout -> ... -> LayerNorm -> lambda x:self.src_attn(x, x, m, src_mask) -> dropout -> ... ->LayerNorm -> self.feed_ward -> dropout
# self_attn: Q = K = V
# src_attn: Q！= K = V （输入不同）
class DecoderLayer(nn.Moudle):
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

# 注意Decoder是由N个DecoderLayer堆叠而成的，使用callable可以直接调用Decoder.forward
class Decoder(nn.moudle):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class EncoderDecoder(nn.Module): 
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        初始化函数中包含五个参数, 分别为编码器对象, 解码器对象, 源数据嵌入数据, 目标数据嵌入数据, 输出生成器
        """
        super(EncoderDecoder, self).__init__() 
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed  #中文的Embedding
        self.tgt_embed = tgt_embed  #英文的Embedding
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):   
        memory = self.encode(src, src_mask)
        res = self.decode(memory, tgt, tgt_mask)
        return res

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src),src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)    

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