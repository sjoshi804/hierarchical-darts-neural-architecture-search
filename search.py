# External Imports
import torch
import torch.nn as nn
import os 

# Internal Imports
from model import ModelController
from operations import OPS, LEN_OPS, SIMPLE_OPS, LEN_SIMPLE_OPS
from util import get_data, save_checkpoint, accuracy, AverageMeter

dir_path = os.getcwd()
# FIXME: SET TO REASONABLE VALUES

# DATASET Config
DATASET = "mnist"
DATAPATH = os.path.join(dir_path, "data/mnist")

# WEIGHTS Config
WEIGHTS_LR = .01
WEIGHTS_LR_MIN = .00001
WEIGHTS_MOMENTUM = 1.
WEIGHTS_WEIGHT_DECAY = 1.
WEIGHTS_GRADIENT_CLIP = 1

# TRAINING CONFIG
EPOCHS = 10
BATCH_SIZE = 1000

# ALPHA Optimizer Config
ALPHA_WEIGHT_DECAY = 1
ALPHA_LR = .01

# HDARTS Config
NUM_LEVELS = 3
NUM_NODES_AT_LEVEL = { 0: 3, 1: 3, 2: 3}
NUM_OPS_AT_LEVEL = { 0: LEN_OPS, 1: 3, 2: 3}
CHANNELS_START = 3
STEM_MULTIPLIER = 3

# MISCELLANEOUS CONFIG
NUM_DOWNLOAD_WORKERS = 2
PRINT_STEP_FREQUENCY = 1
CHECKPOINT_PATH = os.path.join(dir_path, "checkpoints")

class HDARTS:
    def __init__(self):
        self.num_levels = NUM_LEVELS

    def run(self):
        # Get Data & MetaData
        input_size, input_channels, num_classes, train_data = get_data(
            dataset_name=DATASET,
            data_path=DATAPATH,
            cutout_length=0,
            validation=False)

        # Set Loss Criterion
        loss_criterion = nn.CrossEntropyLoss()

        # Initialize model
        model = ModelController(
            num_levels=NUM_LEVELS,
            num_nodes_at_level=NUM_NODES_AT_LEVEL,
            num_ops_at_level=NUM_OPS_AT_LEVEL,
            primitives=OPS,
            channels_in=input_channels,
            channels_start=CHANNELS_START,
            stem_multiplier=1,
            num_classes=num_classes,
            loss_criterion=loss_criterion
        )

        # Weights Optimizer
        w_optim = torch.optim.SGD(
            params=model.get_weights(),
            lr=WEIGHTS_LR,
            momentum=WEIGHTS_MOMENTUM,
            weight_decay=WEIGHTS_WEIGHT_DECAY)

        # Alpha Optimizer - one for each level
        alpha_optim = []
        for level in range(0, NUM_LEVELS):
            alpha_optim.append(torch.optim.Adam(
                    params=[model.get_alpha_level(level)],
                    lr=ALPHA_LR,
                    betas=(0.5, 0.999),
                    weight_decay=ALPHA_WEIGHT_DECAY))


        # Train / Validation Split
        n_train = len(train_data)
        split = n_train // 2
        indices = list(range(n_train))
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
        train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=BATCH_SIZE,
                                                sampler=train_sampler,
                                                num_workers=NUM_DOWNLOAD_WORKERS,
                                                pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=BATCH_SIZE,
                                                sampler=valid_sampler,
                                                num_workers=NUM_DOWNLOAD_WORKERS,
                                                pin_memory=True)
        
        # Learning Rate scheudler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            w_optim, EPOCHS, eta_min=WEIGHTS_LR_MIN)

        # Training Loop
        best_top1 = 0.
        for epoch in range(EPOCHS):
            lr_scheduler.step()

            # TODO: PRINT This?
            lr = lr_scheduler.get_last_lr()

            # TODO: Log alpha_i for each level i
            for level in range(0, self.num_levels):
                print(level, model.alpha[level])

            # training
            self.train(
                train_loader=train_loader,
                valid_loader=valid_loader,
                model=model,
                w_optim=w_optim,
                alpha_optim=alpha_optim,
                epoch=epoch)

            # validation
            cur_step = (epoch+1) * len(train_loader)
            top1 = self.validate(
                valid_loader=valid_loader,
                model=model,
                epoch=epoch,
                cur_step=cur_step)

            # save
            if best_top1 < top1:
                best_top1 = top1
                is_best = True
            else:
                is_best = False

            print("Saving checkpoint")
            save_checkpoint(model, epoch, CHECKPOINT_PATH, is_best)

        print("Final best Prec@1 = {:.4%}".format(best_top1))


    def train(self, train_loader, valid_loader, model: ModelController, w_optim, alpha_optim, epoch):
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()

        cur_step = epoch*len(train_loader)

        # Prepares the model for training - 'training mode'
        model.train()

        for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
            N = trn_X.size(0)

            # Alpha Gradient Steps for each level
            for level in range(0, self.num_levels):
                alpha_optim[level].zero_grad()
                logits = model(trn_X)
                loss = model.loss_criterion(logits, trn_y)
                loss.backward()
                alpha_optim[level].step()

            # Weights Step
            w_optim.zero_grad()
            logits = model(trn_X)
            loss = model.loss_criterion(logits, trn_y)
            loss.backward()

            # gradient clipping
            nn.utils.clip_grad_norm_(model.get_weights(), WEIGHTS_GRADIENT_CLIP)
            w_optim.step()

            prec1, prec5 = accuracy(logits, trn_y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % PRINT_STEP_FREQUENCY == 0 or step == len(train_loader)-1:
                print(
                    "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch+1, EPOCHS, step, len(train_loader)-1, losses=losses,
                        top1=top1, top5=top5))

            print('train/loss', loss.item(), cur_step)
            print('train/top1', prec1.item(), cur_step)
            print('train/top5', prec5.item(), cur_step)
            cur_step += 1

        print("Train: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, EPOCHS, top1.avg))


    def validate(self, valid_loader, model, epoch, cur_step):
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()

        model.eval()

        with torch.no_grad():
            for step, (X, y) in enumerate(valid_loader):
                N = X.size(0)

                logits = model(X)
                loss = model.loss_criterion(logits, y)

                prec1, prec5 = accuracy(logits, y, topk=(1, 5))
                losses.update(loss.item(), N)
                top1.update(prec1.item(), N)
                top5.update(prec5.item(), N)

                if step % PRINT_STEP_FREQUENCY == 0 or step == len(valid_loader)-1:
                    print(
                        "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                        "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                            epoch+1, EPOCHS, step, len(valid_loader)-1, losses=losses,
                            top1=top1, top5=top5))

        print('val/loss', losses.avg, cur_step)
        print('val/top1', top1.avg, cur_step)
        print('val/top5', top5.avg, cur_step)

        print("Valid: [{:2d}/{}] Final Prec@1 {:.4%}".format(epoch+1, EPOCHS, top1.avg))

        return top1.avg

if __name__ == "__main__":
    nas = HDARTS()
    nas.run()