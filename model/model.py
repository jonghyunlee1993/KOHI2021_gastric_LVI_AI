import timm
import torchmetrics
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

class ImageClassifier(pl.LightningModule):
    def __init__(self, model_name, learning_rate, num_classes=3):
        super(ImageClassifier, self).__init__()
        self.model = timm.create_model(model_name, num_classes=num_classes, pretrained=True)
        self.learning_rate = learning_rate
        
        self.train_accuracy = torchmetrics.Accuracy()
        self.valid_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        self.log("train_accuracy", self.train_accuracy(y_hat, y), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("valid_loss", loss)
        self.log("valid_accuracy", self.valid_accuracy(y_hat, y), on_step=False, on_epoch=True, prog_bar=True, logger=True)

    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_accuracy", self.test_accuracy(y_hat, y), on_step=False, on_epoch=True, prog_bar=True, logger=True)
          
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "valid_loss"}

    
def define_callbacks(patience, ckpt_path):
    
    return [EarlyStopping('valid_loss', patience=patience),
            ModelCheckpoint(monitor=ckpt_path)]
