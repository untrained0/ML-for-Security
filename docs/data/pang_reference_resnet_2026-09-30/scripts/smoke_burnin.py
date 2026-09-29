# smoke_burnin.py — CPU smoke test only (pang_reference): the reference train_cifar.py's model, data,
# optimiser and checkpoint format, but stopped after --batches mini-batches, so feder_runner.py has a
# checkpoint to load without a 40-epoch burn-in. The real burn-in is train_cifar.py, unmodified.
import argparse, os, torch, torch.nn as nn
from utils.misc import my_makedir
from utils.train_helpers import prepare_train_data
from utils.model import resnet18

p = argparse.ArgumentParser()
p.add_argument('--batches', default=3, type=int)
p.add_argument('--batch_size', default=32, type=int)
p.add_argument('--workers', default=0, type=int)
p.add_argument('--outf', default='smoke_ckpt')
args = p.parse_args()
my_makedir(args.outf)
net = torch.nn.DataParallel(resnet18(num_classes=10))
_, trloader = prepare_train_data(args, shuffle=True)
opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
crit = nn.CrossEntropyLoss()
net.train()
for i, (x, y) in enumerate(trloader):
    if i >= args.batches: break
    opt.zero_grad(); loss = crit(net(x), y); loss.backward(); opt.step()
    print('batch', i, 'loss', round(loss.item(), 4))
torch.save({'epoch': 0, 'args': args, 'err_cls': 1.0, 'optimizer': opt.state_dict(), 'net': net.state_dict()},
           os.path.join(args.outf, 'epoch40.pth'))
print('saved', os.path.join(args.outf, 'epoch40.pth'))
