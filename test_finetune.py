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
#   data.json is used by memory loader for dataset processing (i.e. memory_loader -> [loaders of datasets]) to extract the pId and cId
# - labels.json is used for applying herding selection over labels.
# - @TODO: weight_directory , representative_memory_directory comes from environment variables


def run_reid():
    weight_directory = "log/resnet50/model/model.pth.tar-1"
    representative_memory_directory = "reid-data/representative-memory"
    should_update: bool = os.path.exists(weight_directory) and os.path.exists(
        representative_memory_directory
    )

    source_dataset_name = "market1501"
    datamanager = torchreid.data.ImageDataManager(
        root="reid-data",
        sources=["chunks", "representative_memory"],
        targets="chunks",
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

        engine.run(
            save_dir="log/resnet50",
            max_epoch=3,
            eval_freq=10,
            print_freq=2,
            test_only=False,
            fixbase_epoch=5,
            open_layers="classifier",
        )
    else:
        engine.run(
            save_dir="log/resnet50",
            max_epoch=3,
            eval_freq=10,
            print_freq=2,
            test_only=False,
        )

    update_representative_memory(
        train_loader=datamanager.train_loader,
        current_dataset_name=source_dataset_name,
        representative_memory_main_directory=representative_memory_directory,
        label_start_index=0,
        label_end_index=3,
        selection_percent=0.5,
        retain_percent=0.5,
    )


if __name__ == "__main__":
    run_reid()
