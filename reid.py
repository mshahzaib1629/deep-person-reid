import torchreid
import os
from torchsummary import summary
from representative_memory import update_representative_memory


def run_reid(
    source_dataset_name,
    source_datasets,
    target_datasets,
    weight_directory,
    rp_memory_dir,
    label_start_index,
    label_end_index,
):
    is_weight_dir_exist = isinstance(weight_directory, str) and os.path.exists(
        weight_directory
    )

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

    if is_weight_dir_exist:
        torchreid.utils.load_pretrained_weights(model, weight_directory)
        print("=> Updating Pre-trained Model")
        engine.run(
            save_dir="log/resnet50",
            max_epoch=10,
            eval_freq=10,
            print_freq=2,
            test_only=False,
            fixbase_epoch=10,
            open_layers="classifier",
            use_early_stopping=True,
            patience=2,
            desired_accuracy=0.30
        )
    else:
        print("=> Training A New Model")
        engine.run(
            save_dir="log/resnet50",
            max_epoch=2,
            eval_freq=10,
            print_freq=2,
            test_only=False,
            use_early_stopping=True,
            patience=2,
            desired_accuracy=0.30
        )

    if isinstance(rp_memory_dir, str):
        update_representative_memory(
            train_loader=datamanager.train_loader,
            current_dataset_name=source_dataset_name,
            representative_memory_main_directory=rp_memory_dir,
            label_start_index=label_start_index,
            label_end_index=label_end_index,
            selection_percent=0.5,
            retain_percent=0.5,
        )
