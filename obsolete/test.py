import torchreid
from torchsummary import summary
from torchreid import utils
from representative_memory import (
    update_representative_memory,
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

    model = torchreid.models.build_model(
        name="resnet50",
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=True,
    )

    # model = model.cuda()

    # num_params, flops = utils.compute_model_complexity(model, (1, 3, 256, 128), verbose=True)

    # print('num_params: ', num_params)
    # print('flops: ', flops)
    # print('model summary: ')
    # print(summary(model, (3, 128, 256), 32))
    # return

    optimizer = torchreid.optim.build_optimizer(model, optim="adam", lr=0.0003)

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
    )

    # extract_images_from_loader(datamanager.train_loader)
    update_representative_memory(
        train_loader=datamanager.train_loader,
        representative_memory_directory="./reid-data/representative-memory",
        label_start_index=0,
        label_end_index=3,
        selection_percent=0.5,
        retain_percent=0.5
    )


if __name__ == "__main__":
    run_reid()
    # image_paths, labels = get_representative_images("./reid-data/representative-memory")
    # for i in range(len(image_paths)):
    #     print(f"{image_paths[i]} => {labels[i]}\n")