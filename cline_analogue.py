import numpy as np


def two_block_inverse(block1: np.ndarray, block2: np.ndarray):
    if block1.shape[0] != block2.shape[0]:
        raise Exception('Different height of blocks!')

    pinv_first_row_height = block1.shape[1]
    pinv_second_row_height = block2.shape[1]

    pblock1 = np.linalg.pinv(block1)
    pblock2 = np.linalg.pinv(block2)

    # np.matmul(pblock2, block1), np.matmul(pblock1, block2) -- may be computed in parallel
    row1 = np.concatenate((np.eye(pinv_first_row_height), np.matmul(pblock1, block2)), axis=1)
    row2 = np.concatenate((np.matmul(pblock2, block1), np.eye(pinv_second_row_height)), axis=1)
    temp_matrix = np.concatenate((row1, row2), axis=0)

    return np.matmul(np.linalg.inv(temp_matrix), np.concatenate((pblock1, pblock2), axis=0))


def two_block_inverse_with_pinv(block1: np.ndarray, pblock1: np.ndarray, block2: np.ndarray, pblock2: np.ndarray):
    if block1.shape[0] != block2.shape[0]:
        raise Exception('Different height of blocks!')

    pinv_first_row_height = block1.shape[1]
    pinv_second_row_height = block2.shape[1]

    row1 = np.concatenate((np.eye(pinv_first_row_height), np.matmul(pblock1, block2)), axis=1)
    row2 = np.concatenate((np.matmul(pblock2, block1), np.eye(pinv_second_row_height)), axis=1)
    temp_matrix = np.concatenate((row1, row2), axis=0)

    return np.matmul(np.linalg.inv(temp_matrix), np.concatenate((pblock1, pblock2), axis=0))


# test: two_block_inverse
matrix = np.random.rand(12, 6)
pseudo_matrix = np.linalg.pinv(matrix)

block1 = matrix[:, :4]
block2 = matrix[:, -2:]
pseudo_by_blocks = two_block_inverse(block1, block2)

assert np.allclose(pseudo_matrix, pseudo_by_blocks)

# test: two_block_inverse_with_pinv
pblock1 = np.linalg.pinv(block1)
pblock2 = np.linalg.pinv(block2)

pseudo_by_blocks_with_pinv = two_block_inverse_with_pinv(block1, pblock1, block2, pblock2)
assert np.allclose(pseudo_matrix, pseudo_by_blocks_with_pinv)