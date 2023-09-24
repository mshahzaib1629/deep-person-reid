from .market1501 import market1501_processor
from .cuhk03 import cuhk03_processor
from .dukemtmc import dukemtmc_processor

processsors = {
    "market1501": market1501_processor,
    "cuhk03": cuhk03_processor,
    "dukemtmc": dukemtmc_processor,
}
