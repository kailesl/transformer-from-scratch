import torch.nn as nn
import torch.nn.functional as F
import torch as torch
import torch
import torch.nn as nn
import math

class PositionEncoder(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len].unsqueeze(0)
#class PositionEncoder:
#   def __init__(self,max_len,d_model):
#        self.d_model=d_model
#        self.max_len=max_len
#        self.p=torch.zeros(max_len,self.d_model)
#        for i in range(max_len):
#            for j in range(self.d_model):
#                if j%2==0:
#                    self.p[i][j]=torch.sin(i/(10000**(j/self.d_model)))
#                else:
#                    self.p[i][j]=torch.cos(i/(10000**(j/self.d_model)))
        
#    def forward(self,x):#x=(max_len,d_model)
#        seq_len=x.shape[1]
#        return x+self.p[:seq_len].unsqueeze(0)
    
class MaskMultiHeadAttention(nn.Module):
    def __init__(self,num_head,d_model):
        super().__init__()
        assert d_model%num_head==0
        self.d_k=d_model//num_head
        self.num_head=num_head
        self.d_model=d_model
        
        self.q=nn.Linear(d_model,d_model)#如果把batch_size也定了就不好搞了，太大了
        self.k=nn.Linear(d_model,d_model)
        self.v=nn.Linear(d_model,d_model)
        self.out=nn.Linear(d_model,d_model)
        self.dropout=nn.Dropout(0.1)
        
    def forward(self,x,mask):#x=(batch_size,max_len,d_model)  mask=(mask_len,max_len)
        batch_size,max_len,_=x.size()
        Q=self.q(x)
        K=self.k(x)#k=(batch_size,max_len,d_model)
        V=self.v(x)
        
        Q=Q.view(batch_size,max_len,self.num_head,self.d_k).transpose(1,2)
        K=K.view(batch_size,max_len,self.num_head,self.d_k).transpose(1,2)
        V=V.view(batch_size,max_len,self.num_head,self.d_k).transpose(1,2)#(batch,num_head,max_len,d_k)
        
        qk=torch.matmul(Q,K.transpose(2,3))
        qk=qk/(self.d_k**0.5)
        mask=mask.to(x.device).unsqueeze(0).unsqueeze(0)
        qk=qk+mask
        
        score=self.dropout(F.softmax(qk,dim=-1))
        attention_out=torch.matmul(score,V)#(batch,num,d_k)
        
        attention_out=attention_out.transpose(1,2)
        attention=attention_out.contiguous().view(batch_size,max_len,self.d_model)
        o=self.out(attention)
        o=self.dropout(o)
        return o

class FeedForwardNetwork(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.d_model=d_model
        self.d_ff=4*d_model
        self.w1=nn.Linear(d_model,self.d_ff)
        self.w2=nn.Linear(self.d_ff,d_model)
        self.dropout=nn.Dropout(0.1)
        
    def forward(self,x):
        input=self.w1(x)
        output=self.dropout(F.gelu(input))
        o=self.w2(output)
        return o

class gpt_decoder(nn.Module):
    def __init__(self,num_head,d_model):
        super().__init__()
        self.ffn=FeedForwardNetwork(d_model)
        self.mmha=MaskMultiHeadAttention(num_head,d_model)
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        
    def forward(self,x):#(batch_size,seq_len,d_model)
        leno=x.shape[1]
        self.mask=torch.triu(torch.ones(leno,leno,device=x.device)*-1e9,diagonal=1)
        po=x
        mo=self.mmha(po,self.mask)
        o1=self.norm1(po+mo)
        fo=self.ffn(o1)
        output=self.norm2(o1+fo)
        return output

class Transformer(nn.Module):
    def __init__(self,max_len,vocab_size,N,num_head,d_model):
        super().__init__()
        self.out=nn.Linear(d_model,vocab_size,bias=True)
        self.blocks=nn.ModuleList([gpt_decoder(num_head,d_model)for _ in range(N)])
        self.norm=nn.LayerNorm(d_model)
        self.pos=PositionEncoder(max_len,d_model)
        self.emd=nn.Embedding(vocab_size,d_model)
        self.out.weight=self.emd.weight
    
    def forward(self,x):#(batch_size,seq_len,d_model)所有处理都是动态的
        x=self.emd(x)
        x=self.pos.forward(x)
        
        for block in self.blocks:
            x=block(x)
        
        x=self.norm(x)
        out=self.out(x)
        return out