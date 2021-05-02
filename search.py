# External Imports
from datetime import datetime
import pprint
import random
import signal
import sys
import torch
import torch.nn as nn
 
# Internal Imports
from config import SearchConfig
from model_controller import ModelController
from operations import OPS, LEN_OPS
from torch.utils.tensorboard import SummaryWriter
from util import get_data, save_checkpoint, accuracy, AverageMeter, print_alpha
 
config = SearchConfig()
 
class HDARTS:
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
        
        # Print config to logs
        pprint.pprint(hparams)

        # Seed for reproducibility
        torch.manual_seed(0)
        random.seed(0)

    def run(self):
        # Get Data & MetaData
        input_size, input_channels, num_classes, train_data = get_data(
            dataset_name=config.DATASET,
            data_path=config.DATAPATH,
            cutout_length=16,
            validation=False)
 
        # Set Loss Criterion
        loss_criterion = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            loss_criterion = loss_criterion.cuda()
        
        # Ensure num of ops at level 0 = num primitives
        config.NUM_OPS_AT_LEVEL[0] = LEN_OPS 
       
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

        ''' 
        Search - Weight Training and Alpha Training
        '''

        # Initialize weight training model i.e. only 1 op at 2nd highest level
        self.model = ModelController(
            num_levels=config.NUM_LEVELS,
            num_nodes_at_level=config.NUM_NODES_AT_LEVEL,
            num_ops_at_level=config.NUM_OPS_AT_LEVEL,
            primitives=OPS,
            channels_in=input_channels,
            channels_start=config.CHANNELS_START,
            stem_multiplier=config.STEM_MULTIPLIER,
            num_classes=num_classes,
            num_cells=config.NUM_CELLS,
            loss_criterion=loss_criterion,
            writer=self.writer
         )
        
        # Transfer model to GPU
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            # Optimize if possible
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
 
        # Weights Optimizer
        w_optim = torch.optim.SGD(
            params=self.model.get_weights(),
            lr=config.WEIGHTS_LR,
            momentum=config.WEIGHTS_MOMENTUM,
            weight_decay=config.WEIGHTS_WEIGHT_DECAY)
        w_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(w_optim, config.EPOCHS, eta_min=config.WEIGHTS_LR_MIN) 
 

        # Alpha Optimizer - one for each level
        alpha_optim = []
        # If trying to simulate DARTS don't bother with alpha optim for higher level
        if config.NUM_NODES_AT_LEVEL[0] == 2:
            num_levels = 1
        else:
            num_levels = config.NUM_LEVELS
        for level in range(0, num_levels):
            alpha_optim.append(torch.optim.Adam(
                    params=self.model.get_alpha_level(level),
                    lr=config.ALPHA_LR[level],
                    weight_decay=config.ALPHA_WEIGHT_DECAY,
                    betas=config.ALPHA_MOMENTUM))

        # Training Loop
        best_top1 = 0.
        for epoch in range(config.EPOCHS):
            lr = w_lr_scheduler.get_lr()[0]

            # Put into weight training mode - turn off gradient for alpha
            self.model.weight_training_mode()
            
            # Weight Training
            self.train_weights(
                train_loader=train_loader,
                model=self.model,
                w_optim=w_optim,
                epoch=epoch,
                lr=lr)

            # Weight Learning Rate Step 
            w_lr_scheduler.step()

            # GPU Memory Allocated for Model in Weight Sharing Phase   
            if epoch == 0:
                try:
                    print("Weight Training Phase: Max GPU Memory Used",torch.cuda.max_memory_allocated()/(1024*1024*1024), "GB")
                except: 
                    print("Unable to retrieve memory data")
            
            # Turn off gradient for weight params
            self.model.alpha_training_mode()

            # Alpha Training / Validation
            top1 = self.train_alpha(
                valid_loader=valid_loader,
                model=self.model,
                alpha_optim=alpha_optim,
                epoch=epoch,
                lr=lr)

            # Save Checkpoint
            if best_top1 < top1:
                best_top1 = top1
                is_best = True
            else:
                is_best = False
            print("Saving checkpoint")
            save_checkpoint(self.model, epoch, config.CHECKPOINT_PATH + "/" + self.dt_string, is_best)
            
            # GPU Memory Allocated for Model
            if epoch == 0:
                try:
                    print("Alpha Training: Max GPU Memory Used",torch.cuda.max_memory_allocated()/(1024*1024*1024), "GB")
                except:
                    print("Unable to print memory data")
            
        # Log Best Accuracy so far
        print("Final best Prec@1 = {:.4%}".format(best_top1))

        # Terminate
        self.terminate()
 
    def train_weights(self, train_loader, model: ModelController, w_optim, epoch, lr):
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()

        cur_step = epoch*len(train_loader)
        
        # Log LR
        self.writer.add_scalar('train/lr', lr, epoch)

        # Prepares the model for training - 'training mode'
        model.train()

        for step, (trn_X, trn_y) in enumerate(train_loader):
            N = trn_X.size(0)
            if torch.cuda.is_available():
                trn_X = trn_X.cuda(non_blocking=True)
                trn_y = trn_y.cuda(non_blocking=True)

            # Weights Step
            w_optim.zero_grad()
            logits = model(trn_X)
            loss = model.loss_criterion(logits, trn_y)
            loss.backward()

            # gradient clipping
            nn.utils.clip_grad_norm_(model.get_weights(), config.WEIGHTS_GRADIENT_CLIP)
            w_optim.step()
 
            prec1, prec5 = accuracy(logits, trn_y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.PRINT_STEP_FREQUENCY == 0 or step == len(train_loader)-1:
                print(
                    datetime.now(),
                    "Weight Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                       epoch+1, config.EPOCHS, step, len(train_loader)-1, losses=losses,
                        top1=top1, top5=top5))
 
            self.writer.add_scalar('train/loss', loss.item(), cur_step)
            self.writer.add_scalar('train/top1', prec1.item(), cur_step)
            self.writer.add_scalar('train/top5', prec5.item(), cur_step)
            cur_step += 1
 
        print("Weight Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.EPOCHS, top1.avg))

    def train_alpha(self, valid_loader, model: ModelController, alpha_optim, epoch, lr):
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()

        cur_step = epoch*len(valid_loader)
        
        # Log LR
        self.writer.add_scalar('train/lr', lr, epoch)

        # Prepares the model for training - 'training mode'
        model.train()

        for step, (val_X, val_y) in enumerate(valid_loader):
            N = val_X.size(0)
            if torch.cuda.is_available():
                val_X = val_X.cuda(non_blocking=True)
                val_y = val_y.cuda(non_blocking=True)

            # Alpha Gradient Steps for each level
            for level in range(len(alpha_optim)):
                alpha_optim[level].zero_grad()
            logits = model(val_X)
            loss = model.loss_criterion(logits, val_y)
            loss.backward()
            for level in range(len(alpha_optim)):
                alpha_optim[level].step()
 
            prec1, prec5 = accuracy(logits, val_y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % config.PRINT_STEP_FREQUENCY == 0 or step == len(valid_loader)-1:
                print(
                    datetime.now(),
                    "Alpha Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                       epoch+1, config.EPOCHS, step, len(valid_loader)-1, losses=losses,
                        top1=top1, top5=top5))
 
            self.writer.add_scalar('val/loss', losses.avg, cur_step)
            self.writer.add_scalar('val/top1', top1.avg, cur_step)
            self.writer.add_scalar('val/top5', top5.avg, cur_step)
            cur_step += 1
 
        print("Alpha Train (Uses Validation Loss): [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.EPOCHS, top1.avg))
        return top1.avg
 
 
    def validate(self, valid_loader, model, epoch, cur_step):
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()

        model.eval()

        with torch.no_grad():
            for step, (X, y) in enumerate(valid_loader):
                N = X.size(0)

                logits = model(X)
                
                if torch.cuda.is_available():
                    y = y.cuda()   

                loss = model.loss_criterion(logits, y)

                prec1, prec5 = accuracy(logits, y, topk=(1, 5))
                losses.update(loss.item(), N)
                top1.update(prec1.item(), N)
                top5.update(prec5.item(), N)
 
                if step % config.PRINT_STEP_FREQUENCY == 0 or step == len(valid_loader)-1:
                    print(
                        datetime.now(),
                        "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                        "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                            epoch+1, config.EPOCHS, step, len(valid_loader)-1, losses=losses,
                            top1=top1, top5=top5))
 
        self.writer.add_scalar('val/loss', losses.avg, cur_step)
        self.writer.add_scalar('val/top1', top1.avg, cur_step)
        self.writer.add_scalar('val/top5', top5.avg, cur_step)

        print("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, config.EPOCHS, top1.avg))
 
        return top1.avg

    def terminate(self, signal=None, frame=None):
        # Print alpha
        print("Alpha Normal")
        print_alpha(self.model.alpha_normal)
        print("Alpha Reduce")
        print_alpha(self.model.alpha_reduce)
        
        # Pass exit signal on
        sys.exit(0)

 
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print('No GPU Available')
    nas = HDARTS()
    nas.run()
