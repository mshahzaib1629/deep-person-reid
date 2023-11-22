import torchreid, os
from representative_memory import (
    update_representative_memory,
     RepresentativeMemory,
    ChunkLoader,
)

torchreid.data.register_image_dataset("representative_memory", RepresentativeMemory)
torchreid.data.register_image_dataset("chunks", ChunkLoader)

def update_rp_memory(source_datasets, source_dataset_name, rp_memory_dir, label_start_index, label_end_index, selection_percent, retain_percent):
    """This helper function is to explicitly update the representative memory with the given data."""
    datamanager = torchreid.data.ImageDataManager(
        root="reid-data",
        sources=source_datasets,
        targets=source_datasets,
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=["random_flip", "random_crop"],
    )

    update_representative_memory(
            train_loader=datamanager.train_loader,
            current_dataset_name=source_dataset_name,
            representative_memory_main_directory=rp_memory_dir,
            label_start_index=label_start_index,
            label_end_index=label_end_index,
            selection_percent=0.5,
            retain_percent=0.5,
        )
    

if __name__ == "__main__":
    RP_MEMORY_PATH = os.path.join("reid-data", "representative-memory")
    update_rp_memory(source_datasets=["chunks", "representative_memory"], 
                     source_dataset_name="market1501", 
                     rp_memory_dir=RP_MEMORY_PATH, 
                     label_start_index=0, 
                     label_end_index=3, 
                     selection_percent=0.5, 
                     retain_percent=0.5)