# External Imports
from datetime import datetime
import os
import signal
import sys
import torch
import torch.nn as nn
 
# Internal Imports
from config import SearchConfig
from .vae_controller import VAEController
from operations import OPS, LEN_OPS
from torch.utils.tensorboard import SummaryWriter
from util import get_data, save_checkpoint, AverageMeter, print_alpha
 
config = SearchConfig()
 
class VAEHDARTS:
    def __init__(self):
        self.dt_string = datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
        self.writer = SummaryWriter(config.LOGDIR + "/" + config.DATASET +  "/" + str(self.dt_string) + "/")
        self.num_levels = config.NUM_LEVELS

        # Set gpu device if cuda is available
        if torch.cuda.is_available():
            torch.cuda.set_device(config.gpus[0]) 

        # Write config to tensorboard
        hparams = {}
        for key in config.__dict__:
            if type(config.__dict__[key]) is dict or type(config.__dict__[key]) is list:
                hparams[key] = str(config.__dict__[key])
            else:
                hparams[key] = config.__dict__[key]
        self.writer.add_hparams(hparams, {'accuracy': 0})

    def run(self):
        # Get Data & MetaData
        input_size, input_channels, num_classes, train_data = get_data(
            dataset_name=config.DATASET,
            data_path=config.DATAPATH,
            cutout_length=0,
            validation=False)
 
        # Set Loss Criterion
        loss_criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            loss_criterion = loss_criterion.cuda()
        
        # Ensure num of ops at level 0 = num primitives
        config.NUM_OPS_AT_LEVEL[0] = LEN_OPS 
       
        # Initialize vae
        self.vae = VAEController(
            num_levels=config.NUM_LEVELS,
            num_nodes_at_level=config.NUM_NODES_AT_LEVEL,
            num_ops_at_level=config.NUM_OPS_AT_LEVEL,
            primitives=OPS,
            channels_in=input_channels,
            beta=250,
            image_height=input_size[1],
            image_width=input_size[0],
            writer=self.writer
         )

        if torch.cuda.is_available():
            self.vae = self.vae.cuda()
 
        # Weights Optimizer
        w_optim = torch.optim.SGD(
            params=self.vae.get_weights(),
            lr=config.WEIGHTS_LR,
            momentum=config.WEIGHTS_MOMENTUM,
            weight_decay=config.WEIGHTS_WEIGHT_DECAY)
 
        # Alpha Optimizer - one for each level
        alpha_optim = []
        for level in range(0, config.NUM_LEVELS):
            alpha_optim.append(torch.optim.Adam(
                    params=self.vae.get_alpha_level(level),
                    lr=config.ALPHA_LR,
                    weight_decay=config.ALPHA_WEIGHT_DECAY))
 
 
        # Train / Validation Split
        n_train = (len(train_data) // 100) * config.PERCENTAGE_OF_DATA
        split = n_train // 2
        indices = list(range(n_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
        train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=config.BATCH_SIZE,
                                                sampler=train_sampler,
                                                num_workers=config.NUM_DOWNLOAD_WORKERS,
                                                pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=config.BATCH_SIZE,
                                                sampler=valid_sampler,
                                                num_workers=config.NUM_DOWNLOAD_WORKERS,
                                                pin_memory=True)

        # Register Signal Handler for interrupts & kills
        signal.signal(signal.SIGINT, self.terminate)

        # Training Loop
        best_disentanglement = float('inf')
        for epoch in range(config.EPOCHS):
 
            # Training
            self.train(
                train_loader=train_loader,
                valid_loader=valid_loader,
                vae=self.vae,
                w_optim=w_optim,
                alpha_optim=alpha_optim,
                epoch=epoch)

            # Validation
            cur_step = (epoch+1) * len(train_loader)
            disentanglement = self.validate(
                valid_loader=valid_loader,
                vae=self.vae,
                epoch=epoch,
                cur_step=cur_step)

            # Save Checkpoint
            if best_disentanglement < disentanglement:
                best_disentanglement = disentanglement
                is_best = True
            else:
                is_best = False
            print("Saving checkpoint")
            save_checkpoint(self.vae, epoch, config.CHECKPOINT_PATH + "/" + self.dt_string, is_best)
 
        # Log Best Accuracy so far
        print("Final best Disentanglement = {:.4%}".format(best_disentanglement))

        # Terminate
        self.terminate()
 
    def train(self, train_loader, valid_loader, vae: VAEController, w_optim, alpha_optim, epoch):
        losses = AverageMeter()
        disentanglements = AverageMeter()

        cur_step = epoch*len(train_loader)
 
        # Prepares the vae for training - 'training mode'
        vae.train()

        for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
            N = trn_X.size(0)
            if torch.cuda.is_available():
                trn_X = trn_X.cuda()
                trn_y = trn_y.cuda()
                val_X = val_X.cuda()
                val_y = val_y.cuda()

            # Alpha Gradient Steps for each level
            for level in range(0, self.num_levels):
                alpha_optim[level].zero_grad()
                output = vae(trn_X)
                loss = vae.loss(trn_X, output)
                loss.backward()
                alpha_optim[level].step()

            # Weights Step
            w_optim.zero_grad()
            output = vae(trn_X)
            loss = vae.loss(trn_X, output)
            disentanglement = vae.disentanglement(trn_X, output)
            loss.backward()

            # gradient clipping
            nn.utils.clip_grad_norm_(vae.get_weights(), config.WEIGHTS_GRADIENT_CLIP)
            w_optim.step()
 
            # Update averages
            losses.update(loss.item(), N)
            disentanglements.update(disentanglement, N)

            if step % config.PRINT_STEP_FREQUENCY == 0 or step == len(train_loader)-1:
                print(
                    datetime.now(),
                    "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f}"
                    "Disentanglement {disentanglements.avg:.3f}".format(
                       epoch+1, config.EPOCHS, step, len(train_loader)-1, losses=losses, disentanglements=disentanglements))

            self.writer.add_scalar('train/loss', loss.item(), cur_step)
            self.writer.add_scaler('train/disentanglement(kl)', disentanglement, cur_step)
            cur_step += 1
 
        print("Train: [{:2d}/{}] Final Disentanglement {:.4%}".format(epoch+1, config.EPOCHS, disentanglements.avg))
 
 
    def validate(self, valid_loader, vae: VAEController, epoch, cur_step):
        losses = AverageMeter()
        disentanglements = AverageMeter()

        vae.eval()

        with torch.no_grad():
            for step, (X, y) in enumerate(valid_loader):
                N = X.size(0)

                output = vae(X)
                
                if torch.cuda.is_available():
                    y = y.cuda()   

                loss = vae.loss(X, output)
                disentanglement = vae.loss(X, output)

                # update averages
                losses.update(loss.item(), N)
                disentanglements.update(disentanglement, N)
 
                if step % config.PRINT_STEP_FREQUENCY == 0 or step == len(valid_loader)-1:
                    print(
                        datetime.now(),
                        "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                        "Disentanglement {disentanglements.avg:.3f}".format(
                            epoch+1, config.EPOCHS, step, len(valid_loader)-1, losses=losses,
                            disentanglements=disentanglements))
 
        self.writer.add_scalar('val/loss', losses.avg, cur_step)
        self.writer.add_scalar('val/disentanglement', disentanglements.avg, cur_step)

        print("Valid: [{:2d}/{}] Final Disentanglement {:.4%}".format(epoch+1, config.EPOCHS, disentanglements.avg))
 
        return disentanglements.avg

    def terminate(self, signal=None, frame=None):
        # Print alpha
        print_alpha(self.vae.alpha, self.writer)
        
        '''
        TODO: Implement this for VAE
        # Ensure directories to save in exist
        learnt_model_path = config.LEARNT_MODEL_PATH
        if not os.path.exists(learnt_model_path):
            os.makedirs(learnt_model_path)

        # Save learnt model
        learnt_model = LearntModel(self.model.model)
        torch.save(learnt_model, learnt_model_path + "/" + self.dt_string + "_learnt_model")
        '''
        # Pass exit signal on
        sys.exit(0)

 
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print('No GPU Available')
    nas = VAEHDARTS()
    nas.run()
