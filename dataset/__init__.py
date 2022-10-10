import importlib


def find_dataset_ref(dataset_name):
    module = importlib.import_module("dataset." + dataset_name)
    dataset_class = getattr(module, "NVSDataset")
    return dataset_class