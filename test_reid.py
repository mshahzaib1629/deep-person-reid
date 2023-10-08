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


def fresh_train_with_rp(comments="Fresh training with RP"):
    """Train with representative memory without finetuning"""

    worksheet_name = "Fresh with RP"

    source_datasets = ["chunks", "representative_memory"]
    source_dataset_name = "market1501"
    source_dataset_label_start_index = 0
    source_dataset_label_end_index = 3
    target_datasets = "chunks"
    weight_directory = None
    rp_memory_directory = RP_MEMORY_DIR

    print("\n=> Started training with representative memory without finetuning\n")

    run_reid(
        comments,
        worksheet_name,
        source_dataset_name,
        source_datasets,
        target_datasets,
        weight_directory,
        rp_memory_directory,
        use_early_stopping=True,
        epochs=500,
        patience=3,
        desired_accuracy=0.45,
        label_start_index=source_dataset_label_start_index,
        label_end_index=source_dataset_label_end_index,
    )


def finetune_without_rp(comments="Finetuining without RP"):
    """Train with finetuning without representative memory"""

    worksheet_name = "Finetune without RP"

    source_datasets = ["chunks"]
    source_dataset_name = "market1501"
    source_dataset_label_start_index = 0
    source_dataset_label_end_index = 3
    target_datasets = "chunks"
    weight_directory = "log/resnet50/model/model.pth.tar-2"
    rp_memory_directory = None

    print("\n=> Started training with finetuning without representative memory\n")

    run_reid(
        comments,
        worksheet_name,
        source_dataset_name,
        source_datasets,
        target_datasets,
        weight_directory,
        rp_memory_directory,
        use_early_stopping=True,
        epochs=500,
        patience=3,
        desired_accuracy=0.45,
        label_start_index=source_dataset_label_start_index,
        label_end_index=source_dataset_label_end_index,
    )


def finetune_with_rp(comments="Finetuning with RP"):
    """Train with finetuning and representative memory"""

    worksheet_name = "Finetune with RP"

    source_datasets = ["chunks", "representative_memory"]
    source_dataset_name = "market1501"
    source_dataset_label_start_index = 0
    source_dataset_label_end_index = 3
    target_datasets = "chunks"
    weight_directory = "log/resnet50/model/model.pth.tar-2"
    rp_memory_directory = RP_MEMORY_DIR

    print("\n=> Started training with finetuning and representative memory\n")

    run_reid(
        comments,
        worksheet_name,
        source_dataset_name,
        source_datasets,
        target_datasets,
        weight_directory,
        rp_memory_directory,
        use_early_stopping=True,
        epochs=500,
        patience=5,
        desired_accuracy=0.70,
        label_start_index=source_dataset_label_start_index,
        label_end_index=source_dataset_label_end_index,
    )


if __name__ == "__main__":
    COMMENTS = "Trying to add count of new & rp images in google sheet."
    finetune_with_rp(comments=COMMENTS)
