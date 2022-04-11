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


def two_block_inverse_and_put(dict: dict, key: str, block1: np.ndarray, block2: np.ndarray):
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

    result = np.matmul(np.linalg.inv(temp_matrix), np.concatenate((pblock1, pblock2), axis=0))
    dict[key] = result


def two_block_inverse_then_calc_and_put(dict: dict, key: str, block1: np.ndarray, block2: np.ndarray,
                                        multAt: np.ndarray):
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

    result = np.matmul(np.linalg.inv(temp_matrix), np.concatenate((pblock1, pblock2), axis=0))
    dict[key] = result
    if key == '1-2':
        presult = np.concatenate((np.eye(result.shape[0]), np.matmul(result, multAt)), axis=1)
    elif key == '3-4':
        presult = np.concatenate((np.matmul(result, multAt), np.eye(result.shape[0])), axis=1)
    dict['row' + key] = presult


def two_block_inverse_with_pinv(block1: np.ndarray, pblock1: np.ndarray, block2: np.ndarray, pblock2: np.ndarray):
    if block1.shape[0] != block2.shape[0]:
        raise Exception('Different height of blocks!')

    pinv_first_row_height = block1.shape[1]
    pinv_second_row_height = block2.shape[1]

    row1 = np.concatenate((np.eye(pinv_first_row_height), np.matmul(pblock1, block2)), axis=1)
    row2 = np.concatenate((np.matmul(pblock2, block1), np.eye(pinv_second_row_height)), axis=1)
    temp_matrix = np.concatenate((row1, row2), axis=0)

    return np.matmul(np.linalg.inv(temp_matrix), np.concatenate((pblock1, pblock2), axis=0))


def calc_pinv_by_rows_and_inversed(row1, row2, pblock1: np.ndarray, pblock2: np.ndarray):
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

# test: two_block_inverse_and_put
matrix = np.random.rand(12, 6)
pseudo_matrix = np.linalg.pinv(matrix)

block1 = matrix[:, :4]
block2 = matrix[:, -2:]
res = {}
pseudo_by_blocks = two_block_inverse_and_put(res, '12', block1, block2)
assert np.allclose(pseudo_matrix, res['12'])
