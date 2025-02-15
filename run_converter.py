from Converter import *
from Utils import encode_rna, AA_DICT

conv = torch.load("ConverterWeights/converter.pt", map_location=torch.device("cpu"))

seqs = ["AAAAAAAAAA", "GGUAUGAGGGUAUU"]

processed_seqs = [encode_rna(s) for s in seqs] # (batch, seq, base)
lengths = ([len(s) for s in processed_seqs])

# Pad sequences
processed_seqs = torch.nn.utils.rnn.pad_sequence(processed_seqs, batch_first=True, padding_value=0)

# Send sequences through the converter
aa_seqs = conv(processed_seqs) # (seq, batch, aa)
# Reconvert to letter representation
aa_seqs_strings = ["".join(map(AA_DICT.get, aa_seq[:length])) for aa_seq, length in zip(aa_seqs, lengths)]

print(aa_seqs_strings)
