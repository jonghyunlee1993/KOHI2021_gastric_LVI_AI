import torch
from config import *
from utils.data_loader import *
from model.model import *

if __name__ == "__main__":
    positive_flist = glob.glob(DATA_POSITIVE_PATH)
    negative_flist = glob.glob(DATA_NEGATIVE_PATH)
    normal_flist = glob.glob(DATA_NORMAL_PATH)
    
    background_flist = negative_flist + normal_flist
    
    positive_df = generate_patch_df(positive_flist, DATA_POSITIVE_LABEL)
    background_df = generate_patch_df(background_flist, DATA_BACKGROUND_LABEL)

    X_train, X_valid, X_test, y_train, y_valid, y_test = define_dataset_pos_back(positive_df, background_df, sampling_rate=0.2)
    print(f"X train: {X_train.shape}\nX valid: {X_valid.shape}\nX test: {X_test.shape}")
    print(f"y train: {y_train.shape}\ny valid: {y_valid.shape}\ny test: {y_test.shape}")

    train_transforms, valid_transforms = define_augmentation()

    train_dataset = LVIDataset(X_train, y_train, transforms=train_transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, shuffle=True)

    valid_dataset = LVIDataset(X_valid, y_valid, transforms=valid_transforms)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, shuffle=False)

    test_dataset = LVIDataset(X_test, y_test, transforms=valid_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, shuffle=False)

    classifier = ImageClassifier(MODEL_NAME, LEARNING_RATE)
    callbacks = define_callbacks(PATIENCE, CKPT_PATH)
    
    if USE_TPU:
        trainer = pl.Trainer(tpu_cores=8, max_epochs=N_EPOCHS, enable_progress_bar=True, callbacks=callbacks)
    else:
        trainer = pl.Trainer(gpus=1, max_epochs=N_EPOCHS, enable_progress_bar=True, callbacks=callbacks)

    trainer.fit(classifier, train_dataloader, valid_dataloader)

