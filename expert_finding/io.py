import numpy as np
import pkg_resources
import logging
import scipy

logger = logging.getLogger()


def load_dataset(dataset_name):
    npz_file = pkg_resources.resource_filename("expert_finding", 'resources/{0}.npz'.format(dataset_name))
    print(npz_file)
    data = np.load(npz_file, allow_pickle=True)
    data_dict = dict()
    for k in data:
        if len(data[k].shape) == 0:
            data_dict[k] = data[k].flat[0]
        else:
            data_dict[k] = data[k]
        logger.debug(
            f"{k:>10} shape = {str(data_dict[k].shape):<20}  "
            f"type = {str(type(data_dict[k])):<50}  "
            f"dtype = {data_dict[k].dtype}")

    A_da = data_dict["A_da"]
    A_dd = data_dict["A_dd"]
    T = data_dict["T"]
    L_d = data_dict["L_d"]
    L_d_mask = data_dict["L_d_mask"]
    L_a = data_dict["L_a"]
    L_a_mask = data_dict["L_a_mask"]
    tags = data_dict["tags"]

    return A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags


def get_list_of_dataset_names():
    files_names = pkg_resources.resource_listdir("expert_finding", 'resources')
    dataset_names = [f[:-4] for f in files_names if f[-4:] == ".npz"]
    return dataset_names


def load_multi_class_dataset(dataset_name):
    npz_file = pkg_resources.resource_filename("idne", 'resources/{0}.npz'.format(dataset_name))
    data = np.load(npz_file, allow_pickle=True)
    data_dict = dict()
    for k in data:
        if len(data[k].shape) == 0:
            data_dict[k] = data[k].flat[0]
        else:
            data_dict[k] = data[k]
        logger.debug(f"{k:>20} shape={data_dict[k].shape}  type={type(data_dict[k])} dtype={data_dict[k].dtype}")

    texts = data_dict["texts"]
    data_adj = data_dict["adjacency_matrix"]
    adjacency_matrix = make_symetric(data_dict["adjacency_matrix"])
    labels = data_dict["labels"]
    labels_mask = data_dict["labels_mask"]
    return adjacency_matrix, texts, labels, labels_mask


def make_symetric(X):
    X.eliminate_zeros()
    X.sum_duplicates()
    rows, cols = X.nonzero()
    data = X.data
    pairs_set = set()
    pairs_list = list()
    pairs_data = list()
    for i, (r, c) in enumerate(zip(rows, cols)):
        if (r, c) not in pairs_set and (c, r) not in pairs_set:
            pairs_set.add((r, c))
            pairs_list.append((r, c))
            pairs_data.append(data[i])

    new_rows = np.array([val[0] for val in pairs_list], dtype=np.int)
    new_cols = np.array([val[1] for val in pairs_list], dtype=np.int)
    new_data = np.array(pairs_data, dtype=np.float)

    a = scipy.sparse.csr_matrix(
        (
            np.concatenate((new_data, new_data)),
            (np.concatenate((new_rows, new_cols)),
             np.concatenate((new_cols, new_rows)))
        ),
        shape=X.shape)

    a.setdiag(0)
    a.sum_duplicates()
    a.eliminate_zeros()
    return a
