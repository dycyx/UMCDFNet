import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=80, help='epoch number')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--blr', type=float, default=0.001, help='base_learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--testsize', type=int, default=384, help='testing dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate of learning rate')

parser.add_argument('--load', type=str, default='./swin_384.pth')


# parser.add_argument('--train_data_root', type=str, default='/hdd/u202420081200017/code/DSCDNet/RGBT/VT5000/Train', help='the training datasets root')
# parser.add_argument('--val_data_root', type=str, default='/hdd/u202420081200017/code/DSCDNet/RGBT/val', help='the value datasets root')
# parser.add_argument('--test_data_root', type=str, default='/hdd/u202420081200017/code/DSCDNet/RGBT/', help='the test datasets root')

parser.add_argument('--train_data_root', type=str, default='/hdd/u202420081200017/data/UVT/VT5000-Train_unalign', help='the training datasets root')
parser.add_argument('--val_data_root', type=str, default='/hdd/u202420081200017/data/UVT/VT5000-Test_unalign', help='the value datasets root')
parser.add_argument('--test_data_root', type=str, default='/hdd/u202420081200017/data/UVT/', help='the test datasets root')

# parser.add_argument('--train_data_root', type=str, default='/hdd/u202420081200017/data/UVT20K/Train', help='the training datasets root')
# parser.add_argument('--val_data_root', type=str, default='/hdd/u202420081200017/data/UVT20K/Test', help='the value datasets root')
# parser.add_argument('--test_data_root', type=str, default='/hdd/u202420081200017/data/UVT/', help='the test datasets root')


parser.add_argument('--save_path', type=str, default='./res/', help='the path to save models and logs')
parser.add_argument('--test_model', type=str, default='./res/CDFF_epoch_best_withoutCAFA.pth', help='saved model path')
parser.add_argument('--maps_path', type=str, default='./maps/CDFF_epoch_best_withoutCAFA/', help='saved model path')

opt = parser.parse_args()
