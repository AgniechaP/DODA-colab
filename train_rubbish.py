from share import *

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict, create_dataloader
from pytorch_lightning import seed_everything



if __name__ == '__main__':
    # Configs
    config = '/content/drive/MyDrive/DODA/v2/configs/controlnet/DODA_rubbish_cldm_kl_4.yaml'
    logger_freq = 5000
    # max_steps = 80000
    max_steps=10000
    sd_locked = True
    learning_rate = 1e-5
    accumulate_grad_batches = 1
    resume_path = '/content/drive/MyDrive/DODA/v2/models/control-rubbish-ldm.ckpt'
    seed=23

    seed_everything(seed)


    train_dataloader, val_dataloader = create_dataloader(config)

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(config).cpu()

    if resume_path is not None:
        model.load_state_dict(load_state_dict(resume_path, location='cpu'))


    model.sd_locked = sd_locked
    model.learning_rate = learning_rate

    checkpoint_callback = ModelCheckpoint(
        dirpath='/content/drive/MyDrive/DODA/v2/logs/',
        filename='{epoch:02d}-{step}',
        save_weights_only= False,
        save_top_k=1,  # Only save the latest checkpoint
    )

    logger = ImageLogger(batch_frequency=logger_freq)
    # trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger, checkpoint_callback], accumulate_grad_batches=accumulate_grad_batches, max_steps=max_steps)
    trainer = pl.Trainer(
    accelerator="gpu",  # Use GPU for acceleration
    precision=32,  # Set precision to 32-bit floating-point
    callbacks=[logger, checkpoint_callback],  # Add your callbacks
    accumulate_grad_batches=accumulate_grad_batches,  # Accumulate gradients if needed
    max_steps=max_steps  # Set the maximum number of training steps
)


    # Train!
    trainer.fit(model, train_dataloaders = train_dataloader, val_dataloaders = val_dataloader)

