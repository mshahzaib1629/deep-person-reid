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
# - @TODO: While training with any data source, we must need to provide the dataset name to update_representative_memory. It'll be used in formation of data.json
#   data.json will be used by dataloaders (memory_loader -> [loaders of datasets]) to extract the pId and cId
# - @TODO: weight_directory , representative_memory_directory comes from environment variables
# - @TODO: Add a nested directory in representative_memory with name "memory" which will contain all the representative memory images
# and place backups besides memory


def run_reid():
    weight_directory = "log/resnet50/model/model.pth.tar-1"
    representative_memory_directory = "reid-data/representative-memory"
    should_update: bool = os.path.exists(weight_directory) and os.path.exists(
        representative_memory_directory
    )
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
        representative_memory_main_directory=representative_memory_directory,
        label_start_index=0,
        label_end_index=3,
        selection_percent=0.5,
        retain_percent=0.5,
    )


if __name__ == "__main__":
    run_reid()
