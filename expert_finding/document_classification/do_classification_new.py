import expert_finding.document_classification.multi_class_classification_new
# import idne.models.idne
# import idne.models.gvnrt
# import idne.models.lsa
# import idne.models.tadw
# import idne.models.graph2gauss
import logging
import sys
import resource
import pkg_resources
import idne.datasets.io
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger()


def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory


def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 * 0.8, hard))


def main():
    dataset_name = "cora"
    models = {
        # 'lsa': idne.models.lsa.Model(),
        # 'tadw': idne.models.tadw.Model(),
        'idne': idne.models.idne.Model(),
        # 'gvnrt': idne.models.gvnrt.Model(),
        # 'graph2gauss': idne.models.graph2gauss.Model()
    }
    adjacency_matrix, texts, labels, labels_mask = idne.datasets.io.load_multi_class_dataset(dataset_name)
    # A_da, A_dd, T, L_d, L_d_mask, L_a, L_a_mask, tags = idne.datasets.io.load_dataset('dblp')
    # adjacency_matrix, A_dd, texts, labels, labels_mask, L_a, L_a_mask, tags = idne.datasets.io.load_dataset('dblp')
    # adjacency_matrix = A_da @ A_da.T
    # dd = A_da @ A_da.T
    # dd.setdiag(0)
    # self.network = dd + A_dd
    print("loading_dataset")
    for model_name, model in models.items():
        scores = multi_class_classification_new.evaluate(
            model,
            adjacency_matrix,
            texts,
            labels,
            labels_mask,
            [0.9],
            n_trials=1
        )
        print(model_name, scores)


if __name__ == '__main__':
    memory_limit()  # Limitates maximun memory usage to half
    try:
        main()
    except MemoryError:
        sys.stderr.write('\n\nERROR: Memory Exception\n')
        sys.exit(1)
