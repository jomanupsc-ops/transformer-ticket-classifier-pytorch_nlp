import torch
import torch.nn as nn
import torch.nn.functional as F
from step2_embedding import loader,emb_layer,emb_size
import math

class Attention(nn.Module):
    def __init__(self,emb_size):
        super().__init__()
        self.emb_size = emb_size

        self.wq = nn.Linear(emb_size,emb_size,bias=False)
        self.wk = nn.Linear(emb_size,emb_size,bias=False)
        self.wv = nn.Linear(emb_size,emb_size,bias=False)

    def forward(self,x,mask=None):
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)

        result = Q @ K.transpose(-2,-1)
        scores = result /math.sqrt(self.emb_size)

        if mask is not None:
            scores = scores.masked_fill(mask == 0 ,-1e9)

        atten_weight = F.softmax(scores,dim=-1)
        context = atten_weight @ V
        return context,atten_weight

if __name__ == "__main__":            # <-- Sari! Entha space-um illama left-la!
      atten = Attention(emb_size)
      for x, y in loader:
        emb_out = emb_layer(x)
        
        
        pad_mask = (x != 0).unsqueeze(1)
        
        context_out, weights = atten(emb_out, mask=pad_mask)

        print("\n--- STEP 3: SELF-ATTENTION OUTPUT ---")
        print("Input Ticket IDs (First Ticket):", x[0].tolist())
        print("Attention Weights Shape        :", weights.shape)     # (2, 8, 8)
        print("Context Output Shape           :", context_out.shape) # (2, 8, 16)
    
        print("\nFirst Word matha 8 words-ku kudutha Attention (Sum = 1.0):")
        print(torch.round(weights[0][0].detach(), decimals=3).tolist())
        break