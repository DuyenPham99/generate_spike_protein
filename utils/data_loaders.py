import numpy as np
import pandas as pd
from utils import aa_letters


def seq_to_one_hot(sequence, aa_key):
    arr = np.zeros((len(sequence), len(aa_key)))
    for j, c in enumerate(sequence):
        arr[j, aa_key[c]] = 1
    return arr


# trả về mảng 2 chiều với số dòng là len(seqlist) (nếu seqlist là 1 chuỗi, ví dụ: seqlist = "ANHYTREFIIYTFEEGNHYTRDVII")
# số cột là len(alphabet)
def to_one_hot(seqlist, alphabet=aa_letters):
    aa_key = {l: i for i, l in enumerate(alphabet)}
    # {'-': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'X': 20, 'Y': 21}
    if type(seqlist) == str:
        return seq_to_one_hot(seqlist, aa_key)
    else:
        encoded_seqs = []
        for prot in seqlist:
            encoded_seqs.append(seq_to_one_hot(prot, aa_key))
        return np.stack(encoded_seqs)


# def right_pad(seqlist, target_length=None):
#     if target_length is None:
#         return seqlist
# assert isinstance(target_length, int), 'Unknown format for argument padding'
# # padded_seqlist = seqlist
# # handle padding either integer or character representations of sequences
# pad_char = '-' if isinstance(seqlist[0], str) else [0] if isinstance(seqlist[0], list) else None
# for seq in seqlist:
#     print(len(seq))
# return [seq + pad_char * (target_length - len(seq)) for seq in seqlist]


# giả sử: target_length = 24
# seqlist = "ANHYTREFIIYTFEEGNHYTRDVII"
# kết quả hàm right_pad là;
# ['A-----------------------',
# 'N-----------------------', 'H-----------------------', 'Y-----------------------', 'T-----------------------',
# 'R-----------------------', 'E-----------------------', 'F-----------------------', 'I-----------------------',
# 'I-----------------------', 'Y-----------------------', 'T-----------------------', 'F-----------------------',
# 'E-----------------------', 'E-----------------------', 'G-----------------------', 'N-----------------------',
# 'H-----------------------', 'Y-----------------------', 'T-----------------------', 'R-----------------------',
# 'D-----------------------', 'V-----------------------', 'I-----------------------', 'I-----------------------']

def one_hot_generator(seqlist, batch_size=32, shuffle=True, alphabet=aa_letters):
    if type(seqlist) == pd.Series:
        seqlist = seqlist.values
    if type(seqlist) == list:
        seqlist = np.array(seqlist)

    n = len(seqlist)  # nb proteins
    epoch = 0

    while True:
        # shuffle
        # print('Effective epoch {}'.format(epoch))
        if shuffle:
            # hoán đổi vị trí giữa các phần tử trong mảng
            perm = np.random.permutation(len(seqlist))
            prots = seqlist[perm]
        else:
            prots = seqlist

            # len(prots) = 997
            # batch_size = 32 => 31

        for i in range(len(prots) // batch_size):  # phép chia lấy phần nguyên
            batch = to_one_hot((prots[i * batch_size:(i + 1) * batch_size]),
                               alphabet=alphabet)
            # if conditions is not None:
            #     yield [batch, conds[i * batch_size:(i + 1) * batch_size]], batch
            # else:
            yield batch, batch

            # Yield là một keyworrd trong Python được sử dụng để trả về giá trị của hàm mà không hủy đi trạng thái của các biến trong hàm.

        epoch += 1
