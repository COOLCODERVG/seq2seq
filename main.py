import torch
from models import Encoder, Decoder, Seq2Seq
from inference import greedy_decode
from vocab import INPUT_IDX, IDX_OUTPUT, SOS_IDX, EOS_IDX

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("seq2seq_number2words.pth", map_location=DEVICE)
encoder = Encoder(len(INPUT_IDX), 64, 128).to(DEVICE)
decoder = Decoder(len(IDX_OUTPUT), 64, 128).to(DEVICE)
model = Seq2Seq(encoder, decoder, device=DEVICE).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Number → Words Converter (type 'exit' to quit)")
while True:
    num_str = input("Enter a number: ").strip()
    if num_str.lower() == "exit":
        break
    if not num_str.isdigit():
        print("Please enter only digits 0-9.\n")
        continue
    pred = greedy_decode(model, num_str, INPUT_IDX, IDX_OUTPUT, SOS_IDX, EOS_IDX)
    print(f"{num_str} -> {pred}\n")