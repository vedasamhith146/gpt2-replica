import torch
from src.model import GPT2

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
def main():
    VOCAB_SIZE=50257
    BLOCK_SIZE=1024
    N_EMBD=768
    N_LAYER=12
    N_HEAD=12

    device="cuda" if torch.cuda.is_available() else "cpu"
    print("device:",device)
    model=GPT2(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_layers=N_LAYER,
        n_head=N_HEAD
    ).to(device)

    total_params=count_parameters(model)
    print(f"model parameters: {total_params}")
    B=1
    T=8
    print(f"running forward with B={B},T={T}")
    input_ids=torch.randint(0,VOCAB_SIZE,(B,T),dtype=torch.long,device=device)
    model.eval()
    with torch.no_grad():
        logits=model(input_ids)
    
    print("logits shape:", logits.shape)
    print("logits min/max:",logits.min().item(),logits.max().item())
    print("Done.")

if __name__=="__main__":
    main()
