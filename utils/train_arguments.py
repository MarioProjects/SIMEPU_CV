import os
import argparse
import json


class SmartFormatter(argparse.HelpFormatter):

    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


parser = argparse.ArgumentParser(
    description='SIMEPU Project',
    formatter_class=SmartFormatter)

parser.add_argument('--verbose', action='store_true', help='Verbose mode')
parser.add_argument('--optimizer', type=str, default='sgd',
                    choices=['adam', 'sgd', 'sgd_momentum', 'rmsprop'],
                    help='Training Optimizer')

parser.add_argument('--epochs', type=int, default=150, help='Total number epochs for training')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size for training')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('--min_learning_rate', type=float, default=0.0001, help='Minimum Learning rate - Early stopping')

parser.add_argument('--get_path', action='store_true', help='If dataloaders will return images paths')
parser.add_argument('--weighted_loss', action='store_true', help='Use weighted loss based on class frequency')
parser.add_argument('--plateau_scheduler', action='store_true',
                    help='Use a Plateau LR scheduler to control LR based on Accuracy metric')
parser.add_argument('--steps_best', action='store_true', help='Load best checkpoint when Scheduler Steps')
parser.add_argument('--steps_scheduler', action='store_true',
                    help='Use a LR Steps scheduler to control LR based on epochs')

parser.add_argument('--img_size', type=int, default=256, help='Final img squared size')
parser.add_argument('--crop_size', type=int, default=224, help='Center crop squared size')

parser.add_argument('--model_name', type=str, default='resnet18', help='Model name for training')
parser.add_argument('--pretrained', action='store_true', help='Use Imagenet Pretrained model')
parser.add_argument('--validation_size', type=float, default=0.15, help='Validation set size (Over 1)')

parser.add_argument('--output_dir', type=str, default='results/new_logs+train_info',
                    help='Where progress will be saved')

parser.add_argument('--data_augmentation', action='store_true', help='Apply data augmentations at train time')

try:
    args = parser.parse_args()
except:
    print("Working with Jupyter notebook! (Default Arguments)")
    args = parser.parse_args("")

if args.output_dir == "results/new_logs+train_info":
    args.output_dir = "results/new_logs_{}_{}".format(args.model_name, args.optimizer)

# Create directory
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Save commands in txt - https://stackoverflow.com/a/55114771
with open(args.output_dir+'/commandline_args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
