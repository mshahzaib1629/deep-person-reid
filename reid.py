import torchreid
import os
from torchsummary import summary
from representative_memory import (
    update_representative_memory,
    seperate_new_and_rp_images,
)
from torchreid.utils.worksheet import update_worksheet


def run_reid(
    comments,
    worksheet_name,
    model_name,
    source_dataset_name,
    source_datasets,
    target_datasets,
    weight_directory,
    rp_memory_dir,
    label_start_index,
    label_end_index,
    eval_freq=10,
    epochs=2,
    fixed_epochs=None,
    open_layers=None,
    use_early_stopping=False,
    early_stopping_eval_matric="Rank-5",
    eval_patience=2,
    desired_accuracy=0.30,
    resume_training=False,
):
    can_update = isinstance(weight_directory, str) and os.path.exists(weight_directory)

    metadata = {
        "training_type": "updation" if can_update == True else "fresh",
        "model_used": model_name,
        "source_dataset": source_dataset_name,
        "target_datasets": target_datasets,
        "max_epochs": epochs,
        "eval_freq": eval_freq,
        "used_early_stopping": use_early_stopping,
    }

    if can_update == True:
        metadata["weights_used"] = weight_directory
        
    if fixed_epochs is None:
        fixed_epochs = epochs

    if fixed_epochs != epochs:
        metadata["fixed_epochs"] = fixed_epochs
        metadata["open_layers"] = open_layers

    if use_early_stopping == True:
        metadata["eval_patience"] = eval_patience
        metadata["early_stopping_eval_matric"] = early_stopping_eval_matric
        metadata["desired_accuracy"] = desired_accuracy

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

    new_images, rp_images = seperate_new_and_rp_images(
        datamanager.train_loader, rp_memory_dir
    )
    metadata["images_trained_on"] = {}
    metadata["images_trained_on"]["new"] = len(new_images)
    metadata["images_trained_on"]["rp_memory"] = len(rp_images)

    os.environ["WORKSHEET_CONNECTOR_PATH"] = "./worksheet_connector.json"
    if (
        resume_training == False
        and os.path.exists(os.environ.get("WORKSHEET_CONNECTOR_PATH")) == False
    ):
        # if training is not resumed and temp.json do not exist to target a row in worksheet, create new row entry
        update_worksheet(
            excel_link="https://docs.google.com/spreadsheets/d/1qtLI_GLpcnPONtLXDg56aBfNlp5r1jlSMQ5QORbuBVs/edit?usp=sharing",
            worksheet_name=worksheet_name,
            comments=comments,
            metadata=metadata,
        )

    model = torchreid.models.build_model(
        name=model_name,
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

    if can_update == True and resume_training == False:
        torchreid.utils.load_pretrained_weights(model, weight_directory)
        print("=> Updating Pre-trained Model")
        engine.run(
            save_dir=f"log/{model_name}",
            max_epoch=epochs,
            eval_freq=eval_freq,
            print_freq=2,
            test_only=False,
            fixbase_epoch=fixed_epochs,
            open_layers=open_layers,
            use_early_stopping=use_early_stopping,
            eval_patience=eval_patience,
            early_stopping_eval_matric=early_stopping_eval_matric,
            desired_accuracy=desired_accuracy,
        )
    elif can_update == False and resume_training == False:
        print("=> Training A New Model")
        engine.run(
            save_dir=f"log/{model_name}",
            max_epoch=epochs,
            eval_freq=eval_freq,
            print_freq=2,
            test_only=False,
            fixbase_epoch=fixed_epochs,
            open_layers=open_layers,
            use_early_stopping=use_early_stopping,
            eval_patience=eval_patience,
            early_stopping_eval_matric=early_stopping_eval_matric,
            desired_accuracy=desired_accuracy,
        )
    elif resume_training == True and weight_directory is not None:
        print(f"=> Resume Training from {weight_directory}")
        start_epoch = torchreid.utils.resume_from_checkpoint(
            weight_directory, model, optimizer
        )
        engine.run(
            save_dir=f"log/{model_name}",
            max_epoch=epochs,
            start_epoch=start_epoch,
            eval_freq=eval_freq,
            fixbase_epoch=fixed_epochs,
            open_layers=open_layers,
            print_freq=2,
            test_only=False,
            use_early_stopping=use_early_stopping,
            eval_patience=eval_patience,
            early_stopping_eval_matric=early_stopping_eval_matric,
            desired_accuracy=desired_accuracy,
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

    update_worksheet(session_completed=True)
