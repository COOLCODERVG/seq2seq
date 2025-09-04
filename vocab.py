digit_to_word = {
    '0': 'zero','1': 'one','2': 'two','3': 'three','4': 'four',
    '5': 'five','6': 'six','7': 'seven','8': 'eight','9': 'nine'
}

OUTPUT_TOKENS = ['<pad>', '<sos>', '<eos>'] + [digit_to_word[str(d)] for d in range(10)]
OUTPUT_IDX = {tok: i for i, tok in enumerate(OUTPUT_TOKENS)}
IDX_OUTPUT = {i: tok for tok, i in OUTPUT_IDX.items()}
OUTPUT_PAD_IDX = OUTPUT_IDX['<pad>']
SOS_IDX = OUTPUT_IDX['<sos>']
EOS_IDX = OUTPUT_IDX['<eos>']
OUTPUT_VOCAB_SIZE = len(OUTPUT_TOKENS)

INPUT_TOKENS = ['<pad>'] + [str(d) for d in range(10)]
INPUT_IDX = {tok: i for i, tok in enumerate(INPUT_TOKENS)}
INPUT_PAD_IDX = INPUT_IDX['<pad>']
INPUT_VOCAB_SIZE = len(INPUT_TOKENS)