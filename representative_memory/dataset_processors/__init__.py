import sys
sys.path.append("/home/code/Shahzaib/MS/Thesis/Implementation/deep-person-reid/")
from helpers import SelectedDatasets
from .market1501 import market1501_processor
from .cuhk03 import cuhk03_processor
from .dukemtmcreid import dukemtmcreid_processor

processsors = {
    SelectedDatasets.Market1501: market1501_processor,
    SelectedDatasets.CUHK03: cuhk03_processor,
    SelectedDatasets.DukeMTMC: dukemtmcreid_processor,
}


def is_dataset_processor_available(dataset_name: str) -> bool:
    return processsors.keys().__contains__(dataset_name)
