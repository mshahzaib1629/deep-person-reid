import torchreid
from representative_memory import (
    ChunkLoader,
)
torchreid.data.register_image_dataset("chunks", ChunkLoader)

def test_reid_model(model_name, weight_directory, target_datasets):

    datamanager = torchreid.data.ImageDataManager(
        root="reid-data",
        sources=target_datasets,
        targets=target_datasets,
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=["random_flip", "random_crop"],
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
    torchreid.utils.load_pretrained_weights(model, weight_directory)
    engine.run(test_only=True,)


if __name__ == "__main__":
    model_name = "resnet18"
    weight_directory = "log/resnet18/model/model.pth.tar-60"
    dataset = ["market1501"]
    
    test_reid_model(model_name, weight_directory, dataset)
