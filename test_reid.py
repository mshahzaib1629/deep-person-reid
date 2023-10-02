from reid import run_reid
import torchreid
from representative_memory import (
    RepresentativeMemory,
    ChunkLoader,
)
import os

torchreid.data.register_image_dataset("representative_memory", RepresentativeMemory)
torchreid.data.register_image_dataset("chunks", ChunkLoader)


# ====================== Some IMPORTANT RULES ======================================
# - Excluding representative memory, Training source must be one at a time.
#   data.json is used by memory loader for dataset processing (i.e. memory_loader -> [processors of datasets]) to extract the pId and cId
# - labels.json is used for applying herding selection over labels.

# @TODO: Add logs tracking locally and automate for Google Worksheet
# @TODO: Run tests on larger epochs & datasets

RP_MEMORY_DIR = "reid-data/representative-memory"


def fresh_train_with_rp():
    """Train with representative memory without finetuning"""

    source_datasets = ["chunks", "representative_memory"]
    source_dataset_name = "market1501"
    source_dataset_label_start_index = 0
    source_dataset_label_end_index = 3
    target_datasets = "chunks"
    weight_directory = None
    rp_memory_directory = RP_MEMORY_DIR

    print("\n=> Started training with representative memory without finetuning\n")

    run_reid(
        source_dataset_name,
        source_datasets,
        target_datasets,
        weight_directory,
        rp_memory_directory,
        label_start_index=source_dataset_label_start_index,
        label_end_index=source_dataset_label_end_index,
    )


def finetune_without_rp():
    """Train with finetuning without representative memory"""

    source_datasets = ["chunks"]
    source_dataset_name = "market1501"
    source_dataset_label_start_index = 0
    source_dataset_label_end_index = 3
    target_datasets = "chunks"
    weight_directory = "log/resnet50/model/model.pth.tar-2"
    rp_memory_directory = None

    print("\n=> Started training with finetuning without representative memory\n")

    run_reid(
        source_dataset_name,
        source_datasets,
        target_datasets,
        weight_directory,
        rp_memory_directory,
        label_start_index=source_dataset_label_start_index,
        label_end_index=source_dataset_label_end_index,
    )


def finetune_with_rp():
    """Train with finetuning and representative memory"""

    source_datasets = ["chunks", "representative_memory"]
    source_dataset_name = "market1501"
    source_dataset_label_start_index = 0
    source_dataset_label_end_index = 3
    target_datasets = "chunks"
    weight_directory = "log/resnet50/model/model.pth.tar-2"
    rp_memory_directory = RP_MEMORY_DIR

    print("\n=> Started training with finetuning and representative memory\n")

    run_reid(
        source_dataset_name,
        source_datasets,
        target_datasets,
        weight_directory,
        rp_memory_directory,
        label_start_index=source_dataset_label_start_index,
        label_end_index=source_dataset_label_end_index,
    )


if __name__ == "__main__":
    finetune_with_rp()