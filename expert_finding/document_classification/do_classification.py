# from context import idne

import expert_finding.document_classification.multi_class_classification as clf
import expert_finding.resources
import expert_finding.models.idne
import expert_finding.io
import logging
import numpy as np
import pkg_resources
import os

logger = logging.getLogger()
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory


# def memory_limit():
#     soft, hard = resource.getrlimit(resource.RLIMIT_AS)
#     resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 * 0.75, hard))


def main():
    dataset_name = "cora"
    adjacency_matrix, texts, labels, labels_mask = expert_finding.io.load_multi_class_dataset("cora")
    # adjacency_matrix, A_dd, texts, re_labels, labels_mask, L_a, L_a_mask, tags = datasets.io.load_dataset('dblp')
    # dd = A_da @ A_da.T
    # dd.setdiag(0)
    # print(re_labels.data)

    # labels_coo = re_labels.tocoo()
    # labels = np.zeros(114)
    # for (i, a) in enumerate(labels_coo.row):
    #     labels[a] = labels_coo.col[i]

    # for (i, a) in labels.row:
    # print(labels)

    model = expert_finding.models.idne.Model()


    print("Eval")
    scores = clf.evaluate(
        model,
        adjacency_matrix,
        texts,
        labels,
        labels_mask,
        [0.08],
        n_trials=3
    )

    print(scores)

    print("Plot")
    # model.plot_topics()
    # model.plot_direct_topics()
    # model.plot_words_topics_amplitudes()


main()
# if __name__ == '__main__':
#     memory_limit() # Limitates maximun memory usage to half
#     try:
#         main()
#     except MemoryError:
#         sys.stderr.write('\n\nERROR: Memory Exception\n')
#         sys.exit(1)
