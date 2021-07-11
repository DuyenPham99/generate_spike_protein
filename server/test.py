import numpy as np

seqlist = ["XNJHUYY", "JIIUYTGH"]

seqlist = np.array(seqlist)

batch_size = 32
padding = None
alphabet = ['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W',
            'X', 'Y']


def right_pad(seqlist, target_length=None):
    if target_length is None:
        return seqlist
    assert isinstance(target_length, int), 'Unknown format for argument padding'
    # padded_seqlist = seqlist
    # handle padding either integer or character representations of sequences
    pad_char = '-' if isinstance(seqlist[0], str) else [0] if isinstance(seqlist[0], list) else None
    return [seq + pad_char * (target_length - len(seq)) for seq in seqlist]


def seq_to_one_hot(sequence, aa_key):
    arr = np.zeros((len(sequence), len(aa_key)))
    for j, c in enumerate(sequence):
        arr[j, aa_key[c]] = 1
    return arr


def to_one_hot(seqlist, alphabet=alphabet):
    aa_key = {l: i for i, l in enumerate(alphabet)}
    if type(seqlist) == str:
        return seq_to_one_hot(seqlist, aa_key)
    else:
        encoded_seqs = []
        for prot in seqlist:
            encoded_seqs.append(seq_to_one_hot(prot, aa_key))
        return np.stack(encoded_seqs)


epoch = 0

while True:
    # hoán đổi vị trí giữa các phần tử trong mảng
    perm = np.random.permutation(len(seqlist))
    prots = seqlist[perm]

    # len(prots) = 997
    # batch_size = 32 => 31

    for i in range(len(prots) // batch_size):  # phép chia lấy phần nguyên
        batch = to_one_hot(right_pad(prots[i * batch_size:(i + 1) * batch_size], padding),
                           alphabet=alphabet)
        # if conditions is not None:
        #     yield [batch, conds[i * batch_size:(i + 1) * batch_size]], batch
        # else:
        print(batch)

    # Yield là một keyworrd trong Python được sử dụng để trả về giá trị của hàm mà không hủy đi trạng thái của các biến trong hàm.

    epoch += 1
