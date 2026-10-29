import torch

tickets = [
    "my payment failed",
    "app keeps crashing",
    "forgot my password"
]
labels = [0, 1, 2]

# 1. Tokenization function
def tokenize(text):
    return text.lower().split()

# 2. Vocabulary creation
vocab = {"<PAD>": 0, "<UNK>": 1}
for text in tickets:
    for word in tokenize(text):
        if word not in vocab:
            vocab[word] = len(vocab)

# 3. Fixed sequence length (ellathaையும் 8 words-ku fix panrom)
MAX_LEN = 8

# 4. Encode & Pad Function: Text -> IDs -> Fixed Length Tensor
def encode_and_pad(text):
    tokens = tokenize(text)
    # Vocab-la illana <UNK> (1) kuduthuruvom
    ids = [vocab.get(w, 1) for w in tokens]
    
    # Length 8-ah vida kammiya irundha 0 (<PAD>) add panrom
    if len(ids) < MAX_LEN:
        ids = ids + [0] * (MAX_LEN - len(ids))
    else:
        ids = ids[:MAX_LEN] # 8-ah vida jasthi irundha cut panniruvom
        
    return torch.tensor(ids, dtype=torch.long)

# Test panni paapom
if __name__ == "__main__":
    print("Vocabulary:", vocab)
    sample = "my payment failed"
    print("\nSample Text:", sample)
    print("Encoded Tensor:", encode_and_pad(sample))
