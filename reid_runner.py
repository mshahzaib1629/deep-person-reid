from helpers import SelectedDatasets, Matric, AvailableModels
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

RP_MEMORY_PATH = os.path.join("reid-data", "representative-memory")

def fresh_train_with_rp(comments="Fresh training with RP"):
    """Train with representative memory without finetuning"""

    worksheet_name = "Fresh with RP"

    model_name = AvailableModels.ResNet18

    source_datasets = [SelectedDatasets.Chunks, SelectedDatasets.RP_Memory]
    source_dataset_name = SelectedDatasets.Market1501
    source_dataset_label_start_index = 0
    source_dataset_label_end_index = 3
    target_datasets = SelectedDatasets.Market1501
    weight_directory = None
    rp_memory_directory = RP_MEMORY_PATH

    print("\n=> Started training with representative memory without finetuning\n")

    run_reid(
        comments,
        worksheet_name,
        model_name,
        source_dataset_name,
        source_datasets,
        target_datasets,
        weight_directory,
        rp_memory_directory,
        use_early_stopping=True,
        epochs=500,
        eval_freq=10,
        eval_patience=1,
        early_stopping_eval_matric="Rank-5",
        desired_accuracy=0.70,
        label_start_index=source_dataset_label_start_index,
        label_end_index=source_dataset_label_end_index,
    )


def finetune_without_rp(comments="Finetuining without RP"):
    """Train with finetuning without representative memory"""

    worksheet_name = "Finetune without RP"

    model_name = AvailableModels.ResNet18

    source_datasets = [SelectedDatasets.Chunks]
    source_dataset_name = SelectedDatasets.Market1501
    source_dataset_label_start_index = 0
    source_dataset_label_end_index = 3
    target_datasets = SelectedDatasets.Market1501
    weight_directory =  os.path.join("log", model_name, "model", "finetune-rp-res18-r22-model.pth.tar-40")
    rp_memory_directory = None

    print("\n=> Started training with finetuning without representative memory\n")

    run_reid(
        comments,
        worksheet_name,
        model_name,
        source_dataset_name,
        source_datasets,
        target_datasets,
        weight_directory,
        rp_memory_directory,
        fixed_epochs=20,
        open_layers=["layer4", "classifier"],
        use_early_stopping=True,
        epochs=500,
        eval_freq=10,
        eval_patience=1,
        early_stopping_eval_matric="Rank-5",
        desired_accuracy=0.70,
        label_start_index=source_dataset_label_start_index,
        label_end_index=source_dataset_label_end_index,
        resume_training=False,
    )


def finetune_with_rp(comments="Finetuning with RP"):
    """Train with finetuning and representative memory"""

    worksheet_name = "Finetune with RP - ResNet18"

    model_name = AvailableModels.ResNet18Att

    source_datasets = [SelectedDatasets.Chunks, SelectedDatasets.RP_Memory]
    source_dataset_name = SelectedDatasets.DukeMTMC
    source_dataset_label_start_index = 0
    source_dataset_label_end_index = 3
    target_datasets = [SelectedDatasets.DukeMTMC]
    weight_directory =  os.path.join("log", model_name, "model", "finetune-rp-res18-r24-model.pth.tar-20")
    # weight_directory = None
    rp_memory_directory = RP_MEMORY_PATH


    print("\n=> Started training with finetuning and representative memory\n")

    run_reid(
        comments,
        worksheet_name,
        model_name,
        source_dataset_name,
        source_datasets,
        target_datasets,
        weight_directory,
        rp_memory_directory,
        use_early_stopping=True,
        fixed_epochs=20,
        open_layers=["layer4", "classifier"],
        epochs=20,
        eval_freq=5,
        eval_patience=1,
        early_stopping_eval_matric=Matric.rank5,
        desired_accuracy=0.90,
        label_start_index=source_dataset_label_start_index,
        label_end_index=source_dataset_label_end_index,
        resume_training=False
    )


if __name__ == "__main__":
    COMMENTS = "Applied Attenion in the end of the model (i.e. after feature mapping) instead of at the start (i.e. before the feature mapping)."
    finetune_with_rp(comments=COMMENTS)