import torch
from transformers import AutoTokenizer, T5EncoderModel, T5Model

def main():
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = T5Model.from_pretrained("t5-base")
    x = "I use Huggingface ."
    x = tokenizer(x, return_tensors="pt", padding=False, truncation=False)

    print(x)
    emb = model.encoder.embed_tokens.weight[:10000,:]
    wf = open("test.tt", "w")
    wf.write(str(emb.tolist()))
    wf.close()
    print("done")
if __name__ == "__main__":
    main()
