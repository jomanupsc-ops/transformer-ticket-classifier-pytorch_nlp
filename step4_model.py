import torch
import torch.nn as nn
from step1_data import vocab
from step2_embedding import loader,emb_size
from step3_attention import Attention
class TransformerClassifier(nn.Module):
    def __init__(self,vocab_size,emb_size,max_length = 8 ,num_classes=3):
        super().__init__()
        self.word_emb = nn.Embedding(vocab_size,emb_size,padding_idx=0)
        self.pos_emb = nn.Embedding(max_length,emb_size)
        self.attention = Attention(emb_size)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

        self.ffn = nn.Sequential(
            nn.Linear(emb_size,emb_size*2),
            nn.ReLU(),
            nn.Linear(emb_size*2,emb_size)
        )
        self.classifier = nn.Linear(emb_size,num_classes)
    def forward(self,x):
        batch_size,seq_len = x.shape
        position= torch.arange(seq_len,device=x.device).unsqueeze(0).expand(batch_size,seq_len)
        out = self.word_emb(x)+self.pos_emb(position)
        pad_mask = (x!=0).unsqueeze(1)
        att_out,weights = self.attention(out,mask=pad_mask)
        out = self.norm1(att_out+out)
        ffn_out = self.ffn(out)
        out = self.norm2(out+ffn_out)
        sentence_vector = out.mean(dim = 1)
        logits = self.classifier(sentence_vector)
        return logits, weights
if __name__ == "__main__":
    model = TransformerClassifier(vocab_size=len(vocab),emb_size=emb_size,num_classes=3)
    for x,y in loader:
        logits,weights = model(x)
        print("\n--- STEP 4: TRANSFORMER CLASSIFIER OUTPUT ---")
        print("Input Shape (Batch, Seq_Len)       :", x.shape)       # [2, 8]
        print("Output Logits Shape (Batch, Class) :", logits.shape)  # [2, 3]
        print("\nFirst Ticket Raw Scores (Billing, Bug, Account):")
        print(torch.round(logits[0].detach(), decimals=3).tolist())
        
        # Endha class-ku அதிக score vandhurukku? (argmax)
        predicted_class = torch.argmax(logits, dim=-1)
        print("Predicted Category IDs :", predicted_class.tolist())
        print("Actual Category IDs    :", y.tolist())
        break