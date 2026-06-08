import os
import torch.nn.functional as F
import torch as torch
import random
import time

from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from gpt_decoder import Transformer

from transformers import AutoTokenizer
TOKENIZER_PATH = "/root/autodl-tmp/Qwen2.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
def get_txt_files(folder):
    txt_files=[]
    for file_name in os.listdir(folder):
        if file_name.endswith(".txt"):
            txt_files.append(
                os.path.join(
                    folder,
                    file_name
                )
            )
    return txt_files
#dataset处理+流式切块
class tokenstreamdataset(IterableDataset):
    def __init__(self,tokenizer,files,max_len,stride=128,chunk_size=200000):
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.files=files
        self.stride=stride
        self.chunk_size=chunk_size
        self.buffer_size=10000
    def __iter__(self):
        shuffle_buffer=[]
        token_buffer=[]
        for file in self.files:
            with open(file,"r",encoding="utf-8") as f:
                while True:
                    text=f.read(self.chunk_size)
                    if not text:
                        break
                    tokens=self.tokenizer.encode(text)
                    token_buffer.extend(tokens)
                    while len(token_buffer)>=self.max_len+1:
                        x=token_buffer[:self.max_len]
                        y=token_buffer[1:self.max_len+1]
                        shuffle_buffer.append((x,y))
                        if len(shuffle_buffer)>=self.buffer_size:
                            idx=random.randrange(len(shuffle_buffer))
                            fo,g=shuffle_buffer.pop(idx)
                            yield torch.tensor(fo,dtype=torch.long),torch.tensor(g,dtype=torch.long)
                        token_buffer=token_buffer[self.stride:]
        random.shuffle(shuffle_buffer)
        while shuffle_buffer:
            fo,g=shuffle_buffer.pop()
            yield torch.tensor(fo,dtype=torch.long),torch.tensor(g,dtype=torch.long)
def load_checkpoint(path,model,optimizer,device):
    checkpoint=torch.load(path,map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch=checkpoint["epoch"]
    step=checkpoint["step"]
    print("成功加载checkpoint")
    return epoch,step

def train(epoch_num):
    #超参数设置
    device="cuda" if torch.cuda.is_available() else "cpu"
    max_len=256
    vocab_size=len(tokenizer)
    N=4
    num_head=4
    d_model=256
    model=Transformer(
        max_len,
        vocab_size,
        N,
        num_head,
        d_model
    ).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    scaler=torch.amp.GradScaler("cuda")
    #读取文件夹
    files=get_txt_files(
        "/root/autodl-tmp/gpt流式预训练2026.6.6/txt data"
    )
    #读取checkpoints
    # epoch,step=load_checkpoint(
    #     "/root/autodl-tmp/gpt流式预训练2026.6.6/checkpoints/ckpt_epoch0_step8000.pth",
    #     model,
    #     optimizer,
    #     device
    # )
    #dataset处理
    dataset=tokenstreamdataset(tokenizer,files,max_len)
    loader=DataLoader(dataset,batch_size=16,num_workers=0,pin_memory=True)
    #打印模型参数量
    total_params=sum(p.numel()for p in model.parameters())
    print(total_params)
    
    for epoch in range(epoch_num):
        start=time.time()
        step_loss=0
        epoch_loss=0
        for step,(x,y) in enumerate(loader):
            x=x.to(device)
            y=y.to(device)
            with torch.amp.autocast("cuda"):
                out=model(x)
                
                loss=F.cross_entropy(
                    out.view(-1,vocab_size),
                    y.view(-1),
                    ignore_index=-100
                )
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step_loss+=loss.item()
            epoch_loss+=loss.item()
            if (step+1)%100==0:
                print(f"epoch={epoch} | step={step} | loss={loss.item():.4f} | 100step_loss={step_loss/100:.4f}")
                step_loss=0
                end=time.time()
                print(f"100 step耗时:{end-start:.2f}s")
                start=time.time()
        epoch_loss=epoch_loss/(step+1)
        print(f"epoch_loss={epoch_loss:.4f}")
            
            
        torch.save(
            {
                "epoch":epoch,
                "model":model.state_dict(),
                "optimizer":optimizer.state_dict(),
                "step":step
            },
            f"/root/autodl-tmp/gpt流式预训练2026.6.6/checkpoints/ckpt_epoch{epoch}_step{step}.pth"
        )
train(5)
#注释代码区
#data切块
#def token_process(tokens,max_len):
#    data=[]
#    for i in range(0,len(tokens),max_len//2):
#        chunk=tokens[i:i+max_len]
#        data.append(chunk)
#    return data
 
#def file_read(folder,window_size):
#    with open(folder,"r",encoding="utf-8") as f:
#        text=f.read
        
    # for epoch in range(epoch_num):
    #     for step,(x,y) in enumerate(loader):
    #         x=x.to(device)
    #         y=y.to(device)
            
    #         out=model(x)
            
    #         loss=F.cross_entropy(
    #             out.view(-1,vocab_size),
    #             y.view(-1),
    #             ignore_index=-100
    #         )
            
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         if step%100==0:
    #             print(f"epoch={epoch} | step={step} | loss={loss.item():.4f}")
            
    #         if step%1000==0:
    #             torch.save(
    #                 {
    #                     "epoch":epoch,
    #                     "model":model.state_dict(),
    #                     "optimizer":optimizer.state_dict(),
    #                     "step":step
    #                 },
    #                 f"G:/ai/checkpoints/gpt-13m-200mtoken-2026.6.6/ckpt_epoch{epoch}_step{step}.pth"
    #             )
    
    # #读取文件
    # tokens=[]
    # for i,file in enumerate(files):
    #     with open(file,"r",encoding="utf-8") as f:
    #         text=f.read()
    #         text=tokenizer.encode(text)
    #         tokens.extend(text)
    #         print(f"text{i}")
    # #dataset处理
    # dataset=GPTDataset(tokens,max_len)
    # loader=DataLoader(dataset,batch_size=8,shuffle=True)