# External Imports
from datetime import datetime
import os
import signal
import sys
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Internal Imports
from config import SearchConfig
from vae_controller import VAEController
from operations import VAE_OPS, LEN_VAE_OPS
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
        config.NUM_OPS_AT_LEVEL[0] = LEN_VAE_OPS 
       
        # Initialize vae
        self.model = VAEController(
            num_levels=config.NUM_LEVELS,
            num_nodes_at_level=config.NUM_NODES_AT_LEVEL,
            num_ops_at_level=config.NUM_OPS_AT_LEVEL,
            primitives=VAE_OPS,
            channels_in=input_channels,
            beta=250,
            image_height=input_size,
            image_width=input_size,
            writer=self.writer
         )

        if torch.cuda.is_available():
            self.model = self.model.cuda()
 
        # Weights Optimizer
        w_optim = torch.optim.SGD(
            params=self.model.get_weights(),
            lr=config.WEIGHTS_LR,
            momentum=config.WEIGHTS_MOMENTUM,
            weight_decay=config.WEIGHTS_WEIGHT_DECAY)
 
        # Alpha Optimizer - one for each level
        alpha_optim = []
        for level in range(0, config.NUM_LEVELS):
            alpha_optim.append(torch.optim.Adam(
                    params=self.model.get_alpha_level(level),
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
        best_entanglement = float('inf')
        for epoch in range(config.EPOCHS):
 
            # Training
            self.train(
                train_loader=train_loader,
                valid_loader=valid_loader,
                model=self.model,
                w_optim=w_optim,
                alpha_optim=alpha_optim,
                epoch=epoch)

            # Validation
            cur_step = (epoch+1) * len(train_loader)
            entanglement = self.validate(
                valid_loader=valid_loader,
                model=self.model,
                epoch=epoch,
                cur_step=cur_step)

            # Save Checkpoint
            if best_entanglement > entanglement:
                best_entanglement = entanglement
                is_best = True
            else:
                is_best = False
            print("Saving checkpoint")
            save_checkpoint(self.model, epoch, config.CHECKPOINT_PATH + "/" + self.dt_string, is_best)
 
        # Log Best Accuracy so far
        print("Final best entanglement = {:.4%}".format(best_entanglement))

        # Terminate
        self.terminate()
 
    def train(self, train_loader, valid_loader, model: VAEController, w_optim, alpha_optim, epoch):
        losses = AverageMeter()
        entanglements = AverageMeter()

        cur_step = epoch*len(train_loader)
 
        # Prepares the vae for training - 'training mode'
        model.train()

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
                output = model(trn_X)
                loss = model.loss(trn_X, output)
                loss.backward()
                alpha_optim[level].step()

            # Weights Step
            w_optim.zero_grad()
            output = model(trn_X)
            loss = model.loss(trn_X, output)
            entanglement = model.entanglement(trn_X, output)
            loss.backward()

            # gradient clipping
            nn.utils.clip_grad_norm_(model.get_weights(), config.WEIGHTS_GRADIENT_CLIP)
            w_optim.step()
 
            # Update averages
            losses.update(loss.item(), N)
            entanglements.update(entanglement, N)

            if step % config.PRINT_STEP_FREQUENCY == 0 or step == len(train_loader)-1:
                print(
                    datetime.now(),
                    "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Entanglement {entanglements.avg:.3f}".format(
                       epoch+1, config.EPOCHS, step, len(train_loader)-1, losses=losses, entanglements=entanglements))

            self.writer.add_scalar('train/loss', loss.item(), cur_step)
            self.writer.add_scalar('train/entanglement(kl)', entanglement, cur_step)
            cur_step += 1
 
        print("Train: [{:2d}/{}] Final entanglement {:.3f}".format(epoch+1, config.EPOCHS, entanglements.avg))
 
 
    def validate(self, valid_loader, model: VAEController, epoch, cur_step):
        losses = AverageMeter()
        entanglements = AverageMeter()

        model.eval()

        with torch.no_grad():
            for step, (X, y) in enumerate(valid_loader):
                N = X.size(0)

                output = model(X)
                
                if torch.cuda.is_available():
                    y = y.cuda()   

                loss = model.loss(X, output)
                entanglement = model.entanglement(X, output)

                # update averages
                losses.update(loss.item(), N)
                entanglements.update(entanglement, N)
 
                if step % config.PRINT_STEP_FREQUENCY == 0 or step == len(valid_loader)-1:
                    print(
                        datetime.now(),
                        "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                        "Entanglement {entanglements.avg:.3f}".format(
                            epoch+1, config.EPOCHS, step, len(valid_loader)-1, losses=losses,
                            entanglements=entanglements))
 
        self.writer.add_scalar('val/loss', losses.avg, cur_step)
        self.writer.add_scalar('val/entanglement', entanglements.avg, cur_step)

        print("Valid: [{:2d}/{}] Final entanglement {:.4%}".format(epoch+1, config.EPOCHS, entanglements.avg))
 
        return entanglements.avg

    def terminate(self, signal=None, frame=None):
        # Print alpha
        print_alpha(self.model.alpha, self.writer)
        
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
