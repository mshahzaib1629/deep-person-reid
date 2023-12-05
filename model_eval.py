import os
import torchreid
from representative_memory import (
    ChunkLoader,
)
torchreid.data.register_image_dataset("chunks", ChunkLoader)
from torchreid.utils.worksheet import update_worksheet

def evaluate_models(models, model_class, datasets, worksheet_name = None):

    os.environ["WORKSHEET_CONNECTOR_PATH"] = "./analysis_worksheet_connector.json"
    # Create data loader for given datasets
    datamanager = torchreid.data.ImageDataManager(
        root="reid-data",
        sources=datasets,
        targets=datasets,
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=["random_flip", "random_crop"],
    )

    models_base_dir = os.path.join("log", model_class, "model")

    # Iterate list of model names to evaluate
    for model_name in models:
        # Take base address of models which should be evaluated
        weight_directory = os.path.join(models_base_dir, model_name)
        if os.path.exists(weight_directory) == False:
            print('Model does not exist: ', weight_directory)
            continue

        # Create the worksheet_connector

        update_worksheet(
            excel_link="https://docs.google.com/spreadsheets/d/1qtLI_GLpcnPONtLXDg56aBfNlp5r1jlSMQ5QORbuBVs/edit?usp=sharing",
            worksheet_name=worksheet_name,
            is_analysis=True,
            model_on_analysis=model_name
        )

        # Evaluate each model on the dataset
        model = torchreid.models.build_model(
        name=model_class,
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
        torchreid.utils.load_pretrained_weights(model, weight_directory)
        engine.run(test_only=True,)

        update_worksheet(session_completed=True)

if __name__ == "__main__":
    models = [
        "finetune-rp-res18-r22-model.pth.tar-60",
        "finetune-rp-res18-r24-model.pth.tar-20",
        "finetune-rp-res18-r22-model.pth.tar-61"
        ]
    
    model_class = "resnet18"
    
    datasets = ['market1501', 'dukemtmcreid']

    WORKSHEET_NAME = "Test [Analysis] Finetune with RP - ResNet18"

    evaluate_models(models, model_class, datasets, WORKSHEET_NAME)