import torchreid

from torchsummary import summary
from representative_memory import (
    get_representative_images,
    ChunkLoader,
)

torchreid.data.register_image_dataset("chunks", ChunkLoader)


def run_reid():
    datamanager = torchreid.data.ImageDataManager(
        root="reid-data",
        sources="chunks",
        targets="chunks",
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=["random_flip", "random_crop"],
    )

    # extract_images_from_loader(datamanager.train_loader)

    model = torchreid.models.build_model(
        name="resnet50",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=True,
    )

    # model = model.cuda()

    # print('model summary: ')
    # print(summary(model, (3, 128, 256), 32))
    # return

    optimizer = torchreid.optim.build_optimizer(model, optim="adam", lr=0.0003)

    start_epoch = torchreid.utils.resume_from_checkpoint(
        "log/resnet50/model/model.pth.tar-1", model, optimizer
    )

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
        start_epoch=start_epoch,
        test_only=False,
    )


if __name__ == "__main__":
    run_reid()
    # image_paths, labels = get_representative_images("./reid-data/representative-memory")
    # for i in range(len(image_paths)):
    #     print(f"{image_paths[i]} => {labels[i]}\n")
