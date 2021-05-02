# External Imports
from datetime import datetime
from pprint import pprint
from torch.utils.tensorboard.writer import SummaryWriter
import os
import signal
import sys
import torch 
import torch.nn as nn

# Internal Imports
from config import TrainConfig
from learnt_model import LearntModel
from util import AverageMeter, accuracy, get_data, load_alpha, print_alpha
from operations import OPS

# Get Config
config = TrainConfig()

class Train:
    '''
    This class loads a learnt model from a pytorch saved model and trains it.
    '''
    def __init__(self):       
        # Initialize Tensorboard
        self.dt_string = datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
        self.writer = SummaryWriter(config.LOGDIR + "/" + config.DATASET +  "/" + str(self.dt_string) + "/")

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
    
        # Print config to logs
        pprint(hparams)
        
        # Load best alpha
        self.alpha_normal, self.alpha_reduce = load_alpha(config.ALPHA_DIR_PATH)  

    def run(self):
        # Get Data & MetaData
        input_size, input_channels, num_classes, train_data, valid_data = get_data(
            dataset_name=config.DATASET,
            data_path=config.DATAPATH,
            cutout_length=16,
            validation=True)

        # Train / Validation Split
        n_train = len(train_data)
        n_valid = len(valid_data)
        if config.PERCENTAGE_OF_DATA < 100:
            n_train = (n_train // 100) * config.PERCENTAGE_OF_DATA
            n_valid = (n_valid // 100) * config.PERCENTAGE_OF_DATA
            train_data = train_data[:n_train]
            valid_data = valid_data[:n_valid]
        
        train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=config.BATCH_SIZE,
                                                num_workers=config.NUM_DOWNLOAD_WORKERS,
                                                pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=config.BATCH_SIZE,
                                                num_workers=config.NUM_DOWNLOAD_WORKERS,
                                                pin_memory=True)
        
        # Create Model 
        print("Alpha Normal")
        print_alpha(self.alpha_normal)
        print("Alpha Reduce")
        print_alpha(self.alpha_reduce)
        print("Creating Model from these Alpha\n\n")
        self.model = LearntModel(
            alpha_normal=self.alpha_normal,
            alpha_reduce=self.alpha_reduce,
            num_cells=config.NUM_CELLS,
            channels_in=input_channels,
            channels_start=config.CHANNELS_START,
            stem_multiplier=config.STEM_MULTIPLIER,
            num_classes=num_classes,
            primitives=OPS,
            auxiliary=True            
        )

        # Port model to gpu if availabile
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            # cuDNN optimizations if possible
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

        # Weights Optimizer
        w_optim = torch.optim.SGD(
            params=self.model.parameters(),
            lr=config.WEIGHTS_LR,
            momentum=config.WEIGHTS_MOMENTUM,
            weight_decay=config.WEIGHTS_WEIGHT_DECAY)

        # Learning Rate Scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.EPOCHS, eta_min=config.WEIGHTS_LR_MIN)

        # Register Signal Handler for interrupts & kills
        signal.signal(signal.SIGINT, self.terminate)

        # Number of parameters
        print("# of Parameters", sum(param.numel() for name, param in self.model.named_parameters() if param.requires_grad and "auxiliary" not in name))

        # Training Loop
        best_top1 = 0.
        loss_criterion = nn.CrossEntropyLoss()
        for epoch in range(config.EPOCHS):
            lr = lr_scheduler.get_lr()[0]

            # Training (One epoch)
            self.train(
                train_loader=train_loader,
                model=self.model,
                w_optim=w_optim,
                epoch=epoch,
                lr=lr,
                gradient_clip=config.WEIGHTS_GRADIENT_CLIP,
                epochs=config.EPOCHS,
                loss_criterion=loss_criterion)
            
            # Learning Rate Step
            lr_scheduler.step()

            # Validation (One epoch)
            cur_step = (epoch+1) * len(train_loader)
            top1 = self.validate(
                valid_loader=valid_loader,
                model=self.model,
                epoch=epoch,
                cur_step=cur_step,
                epochs=config.EPOCHS)

            # Save Checkpoint
            # Creates checkpoint directory if it doesn't exist
            if not os.path.exists(config.CHECKPOINT_PATH + "/" + config.DATASET + "/" + self.dt_string):
                os.makedirs(config.CHECKPOINT_PATH + "/" + config.DATASET + "/" + self.dt_string)
            # torch.save(self.model, config.CHECKPOINT_PATH + "/" + config.DATASET + "/" + self.dt_string + "/" + str(epoch) + ".pt")
            if best_top1 < top1:
                best_top1 = top1
                torch.save(self.model, config.CHECKPOINT_PATH + "/" + config.DATASET + "/" + self.dt_string + "/" + "best.pt")
            # GPU Memory Allocated for Model in Weight Sharing Phase   
            if epoch == 0:
                try:
                    print("Learnt Architecture Training: Max GPU Memory Used",torch.cuda.max_memory_allocated()/(1024*1024*1024), "GB")
                except: 
                    print("Unable to retrieve memory data")
 
        # Log Best Accuracy so far
        print("Final best Prec@1 = {:.4%}".format(best_top1))

        self.terminate()
    
    def train(self, train_loader, model, w_optim, epoch, lr, gradient_clip, epochs, loss_criterion):
        
        # Track average top1, top5, loss
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()

        cur_step = epoch*len(train_loader)

        # Log Learning Rate
        self.writer.add_scalar('train/lr', lr, cur_step)

        # Prepares the model for training - 'training mode'
        model.train()

        for step, (trn_X, trn_y) in enumerate(train_loader):
            N = trn_X.size(0)
            if torch.cuda.is_available():
                trn_X = trn_X.cuda(non_blocking=True)
                trn_y = trn_y.cuda(non_blocking=True)

            # Gradient Step
            w_optim.zero_grad()
            logits,logits_aux = model(trn_X)
            
            loss = loss_criterion(logits, trn_y) # Only supports cross entropy loss rn
            loss += loss_criterion(logits_aux, trn_y) * 0.4 # Make this adjustable
            loss.backward()

            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            w_optim.step()
 
            prec1, prec5 = accuracy(logits, trn_y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.PRINT_STEP_FREQUENCY == 0 or step == len(train_loader)-1:
                print(
                    datetime.now(),
                    "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, epochs, step, len(train_loader)-1, losses=losses,
                        top1=top1, top5=top5))
 
            self.writer.add_scalar('train/loss', loss.item(), cur_step)
            self.writer.add_scalar('train/top1', prec1.item(), cur_step)
            self.writer.add_scalar('train/top5', prec5.item(), cur_step)
            cur_step += 1
 
        print("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, epochs, top1.avg))

    def validate(self, valid_loader, model, epoch, cur_step, epochs):
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()

        model.eval()
        with torch.no_grad():
            for step, (X, y) in enumerate(valid_loader):
                # Batch Size
                N = X.size(0)
                
                # Move tensors to cuda
                if torch.cuda.is_available():
                    X = X.cuda(non_blocking=True)
                    y = y.cuda(non_blocking=True)

                logits = model(X)
                
                if torch.cuda.is_available():
                    y = y.cuda()   

                loss = nn.CrossEntropyLoss()(logits, y) # Only supports Cross Entropy Loss

                prec1, prec5 = accuracy(logits, y, topk=(1, 5))
                losses.update(loss.item(), N)
                top1.update(prec1.item(), N)
                top5.update(prec5.item(), N)

                if step % config.PRINT_STEP_FREQUENCY == 0 or step == len(valid_loader)-1:
                    print(
                        datetime.now(),
                        "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                        "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                            epoch+1, epochs, step, len(valid_loader)-1, losses=losses,
                            top1=top1, top5=top5))
 
        self.writer.add_scalar('val/loss', losses.avg, cur_step)
        self.writer.add_scalar('val/top1', top1.avg, cur_step)
        self.writer.add_scalar('val/top5', top5.avg, cur_step)

        print("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, epochs, top1.avg))
 
        return top1.avg

    def terminate(self, signal=None, frame=None):
        # Save learnt model
        torch.save(self.model, config.CHECKPOINT_PATH + "/" + config.DATASET + "/" + self.dt_string + "/" + "last.pt")

        # Pass exit signal on
        sys.exit(0)

if __name__ == '__main__':
    # Check for CUDA
    if not torch.cuda.is_available():
        print('No GPU Available')

    # Train
    train = Train()
    train.run()
    

    
