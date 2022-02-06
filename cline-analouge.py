import numpy as np


def two_block_inverse(block1: np.ndarray, block2: np.ndarray):
    if block1.shape[0] != block2.shape[0]:
        raise Exception('Different height of blocks!')

    pinv_first_row_height = block1.shape[1]
    pinv_second_row_height = block2.shape[1]

    pblock1 = np.linalg.pinv(block1)
    pblock2 = np.linalg.pinv(block2)

    row1 = np.concatenate((np.eye(pinv_first_row_height), np.matmul(pblock1, block2)), axis=1)
    row2 = np.concatenate((np.matmul(pblock2, block1), np.eye(pinv_second_row_height)), axis=1)
    temp_matrix = np.concatenate((row1, row2), axis=0)

    return np.matmul(np.linalg.inv(temp_matrix), np.concatenate((pblock1, pblock2), axis=0))

#
# def n_block_inverse(blocks):
#     if not all(block.shape[0] for block in blocks):
#         raise Exception('Different height of blocks!')
#
#     temp_matrix_width = sum([block.shape[1] for block in blocks])
#
#     pblocks = np.empty([])
#     for block in blocks:
#         np.append(pblocks, np.linalg.pinv(block))
#         # pblocks = np.concatenate([pblocks, np.linalg.pinv(block)], axis=0)
#     print(pblocks)


# if __name__ == "__main__":
#     b1 = np.random.rand(10, 2)
#     b2 = np.random.rand(10, 3)
    # m = np.concatenate((b1, b2), axis=1)
    # pm = np.linalg.pinv(m)
    # pblocks = block_inverse(b1, b2)
    # print(np.allclose(pblocks, pm))
