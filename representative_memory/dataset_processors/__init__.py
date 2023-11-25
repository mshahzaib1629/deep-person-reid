from .market1501 import market1501_processor
from .cuhk03 import cuhk03_processor
from .dukemtmcreid import dukemtmcreid_processor

processsors = {
    "market1501": market1501_processor,
    "cuhk03": cuhk03_processor,
    "dukemtmcreid": dukemtmcreid_processor,
}


def is_dataset_processor_available(dataset_name: str) -> bool:
    return processsors.keys().__contains__(dataset_name)
