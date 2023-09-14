import torchreid

from torchsummary import summary
from representative_memory import (
    apply_herding_selection,
    get_representative_images,
    RepresentativeMemory,
)
torchreid.data.register_image_dataset("representative_memory", RepresentativeMemory)


def run_reid():
    datamanager = torchreid.data.ImageDataManager(
        root="reid-data",
        sources=["representative_memory"],
        targets="market1501Test",
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

    print("model summary before pretained_weights: ")
    print(summary(model, (3, 128, 256), 32))

    torchreid.utils.load_pretrained_weights(model, "log/resnet50/model/model.pth.tar-1")

    print("model summary after pretained_weights: ")
    print(summary(model, (3, 128, 256), 32))
    # return

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, lr_scheduler="single_step", stepsize=20
    )

    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager, model, optimizer=optimizer, scheduler=scheduler, label_smooth=True
    )

    engine.run(
        save_dir="log/resnet50",
        max_epoch=1,
        eval_freq=10,
        print_freq=2,
        test_only=False,
        fixbase_epoch=5,
        open_layers="classifier",
    )


if __name__ == "__main__":
    run_reid()
