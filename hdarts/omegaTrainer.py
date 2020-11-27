import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim
import numpy as np



# TRAINING THE WEIGHTS OF THE NEURAL NETWORK
class OmegaTrainer:


    def main():
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
        model = model.cuda()
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

        #Using Stochastic Gradient Descent as optimization algorithm
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,

            #momentum is a way to reduce the variance of current gradients
            # by averaging past gradients
            momentum=args.momentum,

            #In order to reduce complexity we multiply the sum of squares(in case weights 
            # are negative and positive) with
            # a smaller number(weight decay constant) then we add this to our 
            #loss function. If our loss function increases due to 
            # weights(non-zero complex interactions) then the program
            # will try to minimize that
            # 
            # Loss = MSE(y_hat, y) + wd * sum(w^2)
            weight_decay=args.weight_decay
        )

        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        #Downloading dataset
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

        #Number of training examples
        num_train = len(train_data)
        indices = list(range(num_train))
        # "split" = amount or subset to train
        split = int(np.floor(args.train_portion * num_train))

        #Subsamples image set
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=2)

        #takes the rest of the images im not sampling.  
        # Leon I think this is "validation set"
        valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True, num_workers=2)

        # Cosine annealing schedule: agressively high learning rate to start out with 
        # that drops on a certain schedule to near zero.  I assume to converge faster
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, float(args.epochs), eta_min=args.learning_rate_min)

        #Get my neural architecture
        architect = Architect(model, args)

        #how many times I pass over my training data
        for epoch in range(args.epochs):
            #Adjusts my learning rate
            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info('epoch %d lr %e', epoch, lr)

            #Gets genotype similar to whats seen in genotypes.py
            genotype = model.genotype()
            logging.info('genotype = %s', genotype)

            #Applies a softmax operation over normal cell architecture
            #??Sid?? these should be singular operations correct
            print(F.softmax(model.alphas_normal, dim=-1))

            #Applies a softmax operation over reduction cell architecture
            #??Sid?? these should be singular operations correct
            print(F.softmax(model.alphas_reduce, dim=-1))

            # training
            #Now lets train the network on CIFAR-10 training data
            #Then compare with valid queue to see error
            #??Sid?? Here why are model and architecture different?
            #Is it because model is the weights for CIFAR-10 learning and architecuture
            train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
            logging.info('train_acc %f', train_acc)

            # validation
            #Lets get our validation accuracy to see how we did
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)

            utils.save(model, os.path.join(args.save, 'weights.pt'))



    def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
        objs = utils.AvgrageMeter()

        #Somehow keeps track of accuracy
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        #??Sid?? Im guessing 
        # - step is epoch?
        # - input is training data
        # - target is the answer
        for step, (input, target) in enumerate(train_queue):
            #??Trains the model which should just be training the weights for CIFAR-10
            model.train()#this part is just training image weights
            n = input.size(0)

            #?? What does torch.autograd(Variable) equal?
            #??This seems to get the training data and does something to it

            #From Sid: "Variable" allows us to go to GPU

            
            # get a random minibatch from the search queue with replacement
            #?? This gets the validation data and does something to it(Variable)?

            #From Sid: "Variable" allows us to go to GPU
            input_search, target_search = next(iter(valid_queue))
            
            #??What does this do? Just seems to perform regular backpropagation
            # Is this the critical bilevel optimization
            #From Sid: alpha level of optimization
            #This will need to change
            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

            #Zeros out the gradients again. So we are back at square one
            optimizer.zero_grad()

            #apparently logits are the unnormalized log probabilities
            #or the probability values BEFORE they are softmaxed - which normalizes
            #them to sum to 1
            logits = model(input)# For now we think its training data loss calculations

            #This calculates the cross entropy loss
            loss = criterion(logits, target)

            #Computes the sum of the gradients of the tensors
            loss.backward()

            #Generally used to clip exploding gradients
            #exploding gradients are large error gradients that accumulate
            #and make the model unable to learn
            nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)

            #??I assume this optimizes the network weights
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.data[0], n)
            top1.update(prec1.data[0], n)
            top5.update(prec5.data[0], n)

            if step % args.report_freq == 0:
                logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

            return top1.avg, objs.avg

    """


    """
    def trainCUDA(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
        objs = utils.AvgrageMeter()

        #Somehow keeps track of accuracy
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        #??Sid?? Im guessing 
        # - step is epoch?
        # - input is training data
        # - target is the answer
        for step, (input, target) in enumerate(train_queue):
            #??Trains the model which should just be training the weights for CIFAR-10
            model.train()#this part is just training image weights
            n = input.size(0)

            #?? What does torch.autograd(Variable) equal?
            #??This seems to get the training data and does something to it

            #From Sid: "Variable" allows us to go to GPU

            #For cuda GPU 
            input = Variable(input, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda(async=True)

            # get a random minibatch from the search queue with replacement
            #?? This gets the validation data and does something to it(Variable)?

            #From Sid: "Variable" allows us to go to GPU
            input_search, target_search = next(iter(valid_queue))
            input_search = Variable(input_search, requires_grad=False).cuda()
            target_search = Variable(target_search, requires_grad=False).cuda(async=True)

            #??What does this do? Just seems to perform regular backpropagation
            # Is this the critical bilevel optimization
            #From Sid: alpha level of optimization
            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

            #Zeros out the gradients again. So we are back at square one
            optimizer.zero_grad()

            #apparently logits are the unnormalized log probabilities
            #or the probability values BEFORE they are softmaxed - which normalizes
            #them to sum to 1
            logits = model(input)# For now we think its training data loss calculations

            #This calculates the cross entropy loss
            loss = criterion(logits, target)

            #Computes the sum of the gradients of the tensors
            loss.backward()

            #Generally used to clip exploding gradients
            #exploding gradients are large error gradients that accumulate
            #and make the model unable to learn
            nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)

            #??I assume this optimizes the network weights
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.data[0], n)
            top1.update(prec1.data[0], n)
            top5.update(prec5.data[0], n)

            if step % args.report_freq == 0:
                logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

            return top1.avg, objs.avg

