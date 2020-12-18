# External Imports
from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter
import argparse
import os
import signal
import sys
import torch 
import torch.nn as nn

# Internal Imports
from util import AverageMeter, accuracy, get_data

# Defaults
LOGDIR="finetune"
DATAPATH = "data"
CHECKPOINT_PATH = "checkpoints_train"
DATASET = "mnist"
EPOCHS = 50
BATCH_SIZE = 64
LR = 0.01
LR_MIN = 0.0001
MOMENTUM = 1.
WEIGHT_DECAY = 0.0003
GRADIENT_CLIP = 1.
NUM_DOWNLOAD_WORKERS = 4

class Train:
    '''
    This class loads a learnt model from a pytorch saved model and trains it.
    '''
    def __init__(self, path_to_model=None, learnt_model=None, save_model_path=None):

        # Validate input and load model
        if path_to_model is not None:
            self.path_to_model = path_to_model
            print("Reading model from " + path_to_model)
            self.model = torch.load(path_to_model)
        elif learnt_model is not None:
            self.path_to_model = save_model_path
            print("Taking model from LearntModel instance from input")
            self.model = learnt_model
        else:
            raise ValueError("No valid value passed to path_to_model or learnt_model, no model to train.")

    def run(self, datapath=DATAPATH, dataset=DATASET, logdir=LOGDIR, checkpoint_path=CHECKPOINT_PATH, percentage_train_data=80, epochs=50, batch_size=64, lr=LR, lr_min=LR_MIN, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, gradient_clip=GRADIENT_CLIP):

        # Set GPU Device if available and port model
        if torch.cuda.is_available():
            torch.cuda.set_device(0) 
            self.model = self.model.cuda()
        
        # Initialize Tensorboard
        self.dt_string = datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
        self.writer = SummaryWriter(logdir + "/" + dataset +  "/" + str(self.dt_string) + "/")

        # Write config to tensorboard
        self.writer.add_hparams(
            {"epochs": epochs, "batch_size": batch_size, "lr": lr, "lr_min": lr_min, "momentum": momentum, "weight_decay": weight_decay, "gradient_clip": gradient_clip},
             {'accuracy': 0})

        # Get Data & MetaData
        input_size, input_channels, num_classes, train_data = get_data(
            dataset_name=dataset,
            data_path=datapath,
            cutout_length=0,
            validation=False)

        # Train / Validation Split
        n_train = len(train_data)
        split = n_train // 100 * percentage_train_data
        indices = list(range(n_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
        train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=batch_size,
                                                sampler=train_sampler,
                                                num_workers=NUM_DOWNLOAD_WORKERS,
                                                pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=batch_size,
                                                sampler=valid_sampler,
                                                num_workers=NUM_DOWNLOAD_WORKERS,
                                                pin_memory=True)
                    
        # Weights Optimizer
        w_optim = torch.optim.SGD(
            params=self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay)

        # Learning Rate Scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, epochs, eta_min=lr_min)

        # Register Signal Handler for interrupts & kills
        signal.signal(signal.SIGINT, self.terminate)

        # Training Loop
        best_top1 = 0.
        for epoch in range(epochs):
            
            # Learning Rate Step
            lr_scheduler.step()
            lr = lr_scheduler.get_lr()[0]

            # Training (One epoch)
            self.train(
                train_loader=train_loader,
                model=self.model,
                w_optim=w_optim,
                epoch=epoch,
                lr=lr,
                gradient_clip=gradient_clip,
                epochs=epochs)

            # Validation (One epoch)
            cur_step = (epoch+1) * len(train_loader)
            top1 = self.validate(
                valid_loader=valid_loader,
                model=self.model,
                epoch=epoch,
                cur_step=cur_step,
                epochs=epochs)

            # Save Checkpoint
            # Creates checkpoint directory if it doesn't exist
            if not os.path.exists(checkpoint_path + "/" + dataset + "/" + self.dt_string):
                os.makedirs(checkpoint_path + "/" + dataset + "/" + self.dt_string)
            torch.save(self.model, checkpoint_path + "/" + dataset + "/" + self.dt_string + "/" + str(epoch) + ".pt")
            if best_top1 < top1:
                best_top1 = top1
                torch.save(self.model, checkpoint_path + "/" + dataset + "/" + self.dt_string + "/" + "best.pt")
 
        # Log Best Accuracy so far
        print("Final best Prec@1 = {:.4%}".format(best_top1))

        self.terminate()
    
    def train(self, train_loader, model, w_optim, epoch, lr, gradient_clip, epochs):
        
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
                trn_X = trn_X.cuda()
                trn_y = trn_y.cuda()

            # Gradient Step
            w_optim.zero_grad()
            logits = model(trn_X)
            loss = nn.CrossEntropyLoss()(logits, trn_y) # Only supports cross entropy loss rn
            loss.backward()

            # gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            w_optim.step()
 
            prec1, prec5 = accuracy(logits, trn_y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

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
                N = X.size(0)

                logits = model(X)
                
                if torch.cuda.is_available():
                    y = y.cuda()   

                loss = nn.CrossEntropyLoss()(logits, y) # Only supports Cross Entropy Loss

                prec1, prec5 = accuracy(logits, y, topk=(1, 5))
                losses.update(loss.item(), N)
                top1.update(prec1.item(), N)
                top5.update(prec5.item(), N)
 
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
        if self.path_to_model is not None:
            path_to_model = self.path_to_model + "_trained"
        else:
            path_to_model = self.dt_string + "_trained"
        torch.save(self.model, path_to_model + ".pt")

        # Pass exit signal on
        sys.exit(0)

if __name__ == '__main__':

    # Register arguments
    parser = argparse.ArgumentParser(prog="train.py", description="Train a model learnt by Hierarchical DARTS")
    parser.add_argument("--path_to_model", help="Path for file which stores the model learnt by HDARTS (should end with learnt_model)")
    parser.add_argument('--datapath', default=DATAPATH, help="Path to store dataset")
    parser.add_argument('--dataset', default=DATASET, help='cifar10 / mnist / fashionmnist')
    parser.add_argument('--logdir', default=LOGDIR, help='directory to save tensorboard logs')
    parser.add_argument('--checkpoint_path', default=CHECKPOINT_PATH, help="Path to checkpoints")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to train for.")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--lr', type=float, default=LR, help='lr for weights')
    parser.add_argument('--lr_min', type=float, default=LR_MIN, help='minimum lr for weights')
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='momentum for weights')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY, help='weight decay for weights')
    parser.add_argument('--gradient_clip', type=float, default=GRADIENT_CLIP, help='gradient clipping for weights')

    # Parse arguments
    args = parser.parse_args()
    print(args)

    # Check for CUDA
    if not torch.cuda.is_available():
        print('No GPU Available')

    # Train
    train = Train(path_to_model=args.path_to_model)
    train.run(
        datapath=args.datapath,
        dataset=args.dataset,
        logdir=args.logdir,
        checkpoint_path=args.checkpoint_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_min=args.lr_min,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip)



    

    