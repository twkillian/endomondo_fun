import argparse

def print_args(args):
    for arg in vars(args):
        print(arg, ': ', getattr(args, arg))

def setup_parser():
    parser = argparse.ArgumentParser(prog = 'TCN&CNN', description = 'Convolution approaches for time series forecasting and prediction')
    parser.add_argument('--dataset_path', type=str, default = '/scratch/hdd001/home/dullerud/endomondo')
    parser.add_argument('--task', choices = ['prediction', 'forecasting'], default='forecasting', help = 'Task on which to train model')
    parser.add_argument('--x_vals', nargs='+', type=str, default=None, help='Name of input values used to train model')
    parser.add_argument('--y_vals', nargs='+', type=str, default=None, help='Name of output values used to train model')
    parser.add_argument('--n_output_vals', type=int, default=1, help='Number of values for y values (only used for prediction task)')
    parser.add_argument('--model', choices=['cnn', 'tcn'], default = 'tcn', help = 'Model architecture to use for training')
    parser.add_argument('--n_hidden', type=int, default = 32)
    parser.add_argument('--n_blocks', type=int, default=8, help='# of temporal blocks (default: 4)')
    parser.add_argument('--pooling_strategy', choices=['max', 'avg'], default = 'max', help = 'Pooling strategy to use in model architecture')
    parser.add_argument('--n_epochs', type=int, default = 250, help = 'Number of epochs to train')
    parser.add_argument('--optimizer', type=str, default = 'adam', help = 'Optimizer to use for training model')
    parser.add_argument('--lr', type=float, default = 0.005, help = 'Learning rate to use for training model')
    parser.add_argument('--weight_decay', type=float, default = 0., help = 'Weight decay to use for training model')
    parser.add_argument('--scheduler', type=str, default = None, help = 'Learning rate scheduler to use for optimizer')
    parser.add_argument('--milestones', nargs = '+', type=int, help = 'Milestone for MultistepLR')
    parser.add_argument('--patience', type=int, default = 10, help = 'Patience for ReduceLROnPlateau')
    parser.add_argument('--loss', type=str, default = 'mse', help = 'Loss to use for training model')
    parser.add_argument('--batch_size', type=int, default = 128, help = 'Batch size to use for training model')
    parser.add_argument('--eval_batch_size', type=int, default = 128, help = 'Batch size to use for evaluating model')
    parser.add_argument('--eval_interval', type=int, default = 50, help = 'Interval size for number of epochs to evaluate model')
    parser.add_argument('--checkpoint_interval', type=int, default = 25, help = 'Interval size for number of epochs to save checkpoint model')
    parser.add_argument('--checkpoint_dir', type=str, default = '.', help = 'Checkpoint directory')
    parser.add_argument('--final_save_fpath', type=str, default = 'saved_models/model.tar', help = 'Final filepath in which to save model')
    return parser
