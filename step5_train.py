import torch
import torch.nn as nn
import torch.optim as optim
from step1_data import vocab,encode_and_pad
from step2_embedding import loader,emb_size
from step4_model import TransformerClassifier
CATEGORIES = {0: "Billing", 1: "Bug", 2: "Account"}
model = TransformerClassifier(vocab_size=len(vocab),emb_size=emb_size,num_classes=len(CATEGORIES))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
for epoch in range(1,51):
    total_loss = 0
    for x,y in loader:
        optimizer.zero_grad()
        logits,weights = model(x)
        loss =criterion(logits,y)
        loss.backward()
        optimizer.step()
        total_loss +=loss.item()
        if epoch %10 ==0 or epoch == 1:
            print(f"Epoch {epoch:2d}/50 | Loss:{total_loss:.4f}")
print("\n--- LIVE TICKET PREDICTION TEST ---")
model.eval()

test_tickets = [
    "my payment failed",
    "app keeps crashing",
    "forgot my password"
]
with torch.no_grad():
    for ticket in test_tickets:
        x = encode_and_pad(ticket).unsqueeze(0)
        logits,weights = model(x)
        predicted_class = torch.argmax(logits,dim=1).item()
        print(f"Ticket: '{ticket:18s}' --> Predicted: {predicted_class} ({CATEGORIES[predicted_class]})")

