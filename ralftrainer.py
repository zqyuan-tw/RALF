import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from utils import evaluate
from models.ralf import RALF, Discriminator
from models.loss import FreqLoss, ReconstructionContrastLoss, EmbeddingContrastLoss



class RALFTrainer(object):
    def __init__(self,
                 log_dir,
                 lr,
                 device=torch.device('cpu')) -> None:
        self.log_dir = log_dir
        self.device = device
        
        # model
        self.model = RALF().to(device)
        self.discriminator = Discriminator().to(device)
        
        # optimizer
        self.opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.opt_model = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0)
        
        # objective function
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.freq_loss = FreqLoss().to(device)
        self.rec_contrast = ReconstructionContrastLoss().to(device)
        self.emb_loss = EmbeddingContrastLoss().to(device)

        self.best_loss = float('inf')
        
        self.start_epoch = 0
        
        # loss hyperparameter
        self.lambda_freq = 1
        self.lambda_adv = 0.1
        self.lambda_recc = 0.1
        self.lambda_rec = 0.1
        self.lambda_emb = 0.1
        
        self.load_from_checkpoint()
        
    def load_from_checkpoint(self):
        ckpt_path = os.path.join(self.log_dir, "checkpoint.tar")
        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            self.start_epoch = checkpoint['epoch'] + 1
            
            self.model.load_state_dict(checkpoint['model'])
            self.discriminator.load_state_dict(checkpoint['disc'])
            
            self.opt_model.load_state_dict(checkpoint['opt_model'])
            self.opt_disc.load_state_dict(checkpoint['opt_disc'])
            
            self.emb_loss.load_state_dict(checkpoint['center'])
            
            self.best_loss = checkpoint['loss']
            print(f"Checkpoint found, will resume training from epoch {self.start_epoch}.")
        
    def save_checkpoint(self, epoch):
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'disc': self.discriminator.state_dict(),
            'opt_model': self.opt_model.state_dict(),
            'opt_disc': self.opt_disc.state_dict(),
            'center': self.emb_loss.state_dict(),
            'loss': self.best_loss,
        }, os.path.join(self.log_dir, "checkpoint.tar"))
        
    def fit(self, n_epoch, train_loader, val_loader=None, val_epoch=0, ckpt_epoch=0, save_best=True):
        writer = SummaryWriter(self.log_dir)
        with trange(self.start_epoch, n_epoch, leave=True, desc="Training Progress", unit='epoch') as total_epoch:
            for epoch in total_epoch:
                self.model.train()
                train_loss = self.train_one_epoch(train_loader, epoch, n_epoch)
                writer.add_scalar('Loss/train', train_loss, epoch + 1)
                
                self.model.eval()
                if val_epoch > 0 and val_loader is not None and (epoch + 1) % val_epoch == 0:
                    val_loss, val_auc, val_acc, val_eer = evaluate(self.model, val_loader, self.bce, self.device)
                    writer.add_scalar('Loss/validation', val_loss, epoch + 1)
                    writer.add_scalar('AUC/validation', val_auc, epoch + 1)
                    
                    if save_best and val_loss < self.best_loss:
                        self.best_loss = val_loss
                        print(f"\nSaving the best model with {self.best_loss:.5f} loss at epoch {epoch + 1}.")
                        torch.save(self.model.state_dict(), os.path.join(self.log_dir, "best.pt"))
                        
                if ckpt_epoch > 0 and (epoch + 1) % ckpt_epoch == 0:
                    self.save_checkpoint(epoch)
                        
        torch.save(self.model.state_dict(), os.path.join(self.log_dir, "last.pt"))
                        
        writer.close()
        
        return self.model
        
    def train_one_epoch(self, dataloader, epoch, n_epoch):
        train_loss = 0
        self.model.train()
        self.discriminator.train()
        with tqdm(dataloader, leave=False, desc=f"Epoch {epoch + 1}", unit='batch') as epoch_pbar:
            for images, labels in epoch_pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                disc_loss = self.train_discriminator_batch(images, labels)
                
                clas_loss, rec_loss = self.train_model_batch(images, labels, epoch / n_epoch)
                
                train_loss += clas_loss
                
                epoch_pbar.set_postfix(clas=clas_loss, disc=disc_loss, rec=rec_loss)
        
        self.emb_loss.update_center()
        
        return train_loss / len(dataloader)
                
    def train_discriminator_batch(self, images, labels):
        real_idx = labels.squeeze(1) == 1
        
        self.opt_disc.zero_grad()
        real_logit = self.discriminator(images[real_idx])
        real_loss = self.bce(real_logit, torch.ones((images[real_idx].size(0), 1), device=self.device))
        with torch.no_grad():
            rec_img = self.model.reconstructor(self.model.destructor(images))
        rec_logit = self.discriminator(rec_img)
        rec_loss = self.bce(rec_logit, torch.zeros((images.size(0), 1), device=self.device))
        disc_loss = real_loss + rec_loss
        disc_loss.backward()
        self.opt_disc.step()
        
        return disc_loss.item()
        
    def train_model_batch(self, images, labels, emb_weight):
        real_idx = labels.squeeze(1) == 1
        
        des_img = self.model.destructor(images)
        rec_img, embedding = self.model.reconstructor(des_img, emb=True)
        self.emb_loss.accumulate_real(embedding, real_idx)
        rec_contr_loss = self.rec_contrast(images, rec_img, real_idx)
        rec_logit = self.discriminator(rec_img)
        rec_loss = self.l1(images[real_idx], rec_img[real_idx]) + \
                   self.lambda_adv * self.bce(rec_logit, torch.ones((images.size(0), 1), device=self.device)) + \
                   self.lambda_freq * self.freq_loss(images[real_idx], rec_img[real_idx])
        pred = self.model.classifier(torch.abs(rec_img - images), torch.abs(images - des_img))
        clas_loss = self.bce(pred, labels)
        total_loss = clas_loss + self.lambda_rec * rec_loss + self.lambda_recc * rec_contr_loss + self.lambda_emb * emb_weight * self.emb_loss(embedding, real_idx)
        total_loss.backward()
        self.opt_model.step()
        self.opt_model.zero_grad()
        
        return clas_loss.item(), rec_loss.item()
                
                