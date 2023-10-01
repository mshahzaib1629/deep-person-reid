import torchreid
import os
from torchsummary import summary
from representative_memory import (
    update_representative_memory,
    RepresentativeMemory,
    ChunkLoader,
)

torchreid.data.register_image_dataset("representative_memory", RepresentativeMemory)
torchreid.data.register_image_dataset("chunks", ChunkLoader)

# ====================== Some IMPORTANT RULES ======================================
# - Excluding representative memory, Training source must be one at a time.
#   data.json is used by memory loader for dataset processing (i.e. memory_loader -> [processors of datasets]) to extract the pId and cId
# - labels.json is used for applying herding selection over labels.

# - In this test, Representative Memory is not being used. We are fine tuning the existing model by updating the last classifier layer.


def run_reid(
    source_dataset_name,
    source_datasets,
    target_datasets,
    weight_directory,
    rp_memory_dir,
    label_start_index,
    label_end_index,
):
    should_update: bool = os.path.exists(weight_directory)

    datamanager = torchreid.data.ImageDataManager(
        root="reid-data",
        sources=source_datasets,
        targets=target_datasets,
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=["random_flip", "random_crop"],
    )

    model = torchreid.models.build_model(
        name="resnet50",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=True,
    )

    optimizer = torchreid.optim.build_optimizer(model, optim="adam", lr=0.0003)

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, lr_scheduler="single_step", stepsize=20
    )

    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager, model, optimizer=optimizer, scheduler=scheduler, label_smooth=True
    )

    if should_update:
        torchreid.utils.load_pretrained_weights(model, weight_directory)
        print("=> Updating Pre-trained Model")
        engine.run(
            save_dir="log/resnet50",
            max_epoch=15,
            eval_freq=10,
            print_freq=2,
            test_only=False,
            fixbase_epoch=10,
            open_layers="classifier",
        )
    else:
        print("=> Training A New Model")
        engine.run(
            save_dir="log/resnet50",
            max_epoch=3,
            eval_freq=10,
            print_freq=2,
            test_only=False,
        )


if __name__ == "__main__":
    source_datasets = ["chunks"]
    source_dataset_name = "market1501"
    source_dataset_label_start_index = 0
    source_dataset_label_end_index = 3
    target_datasets = "chunks"
    weight_directory = "log/resnet50/model/model.pth.tar-3-after-third-ter"
    rp_memory_directory = "reid-data/representative-memory"

    run_reid(
        source_dataset_name,
        source_datasets,
        target_datasets,
        weight_directory,
        rp_memory_directory,
        label_start_index=source_dataset_label_start_index,
        label_end_index=source_dataset_label_end_index,
    )
