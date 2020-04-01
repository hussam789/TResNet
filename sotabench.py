import gc

from torchbench.image_classification import ImageNet
import urllib.request
import torch
from torchvision.transforms import transforms
from src.helper_functions.helper_functions import validate, create_dataloader
from src.models import create_model
import argparse

from src.models.tresnet.tresnet import InplacABN_to_ABN
from src.models.utils.fuse_bn import fuse_bn_recursively

parser = argparse.ArgumentParser(description='PyTorch TResNet ImageNet Inference')
parser.add_argument('--val_dir')
parser.add_argument('--model_path')
parser.add_argument('--model_name', type=str, default='tresnet_m')
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--val_zoom_factor', type=int, default=0.875)
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--num_workers', type=int, default=8)

# parsing args
args = parser.parse_args()
# TResNet-M
model_path = './tresnet_m.pth'
model = create_model(args)
state = torch.load(model_path, map_location='cpu')['model']
model.load_state_dict(state, strict=True)
model = InplacABN_to_ABN(model)
model = fuse_bn_recursively(model)
model = model.cuda()
model.eval()

val_bs = args.batch_size
val_tfms = transforms.Compose(
    [transforms.Resize(int(args.input_size / args.val_zoom_factor)),
     transforms.CenterCrop(args.input_size)])
val_tfms.transforms.append(transforms.ToTensor())

print('Benchmarking TResNet-M')

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='TResNet-M',
    paper_arxiv_id='2003.13630',
    input_transform=val_tfms,
    batch_size=288,
    num_workers=args.num_workers,
    num_gpu=1,
    pin_memory=True,
    paper_results={'Top 1 Accuracy': 0.807, 'Top 5 Accuracy': 0.948},
    model_description="Official weights from the author's of the paper."
)

del model
gc.collect()
torch.cuda.empty_cache()

# TResNet-L
args.model_name = 'tresnet_l'
model_path = './tresnet_l.pth'
model = create_model(args)
state = torch.load(model_path, map_location='cpu')['model']
model.load_state_dict(state, strict=True)
model = InplacABN_to_ABN(model)
model = fuse_bn_recursively(model)
model = model.cuda()
model.eval()

val_bs = args.batch_size
val_tfms = transforms.Compose(
    [transforms.Resize(int(args.input_size / args.val_zoom_factor)),
     transforms.CenterCrop(args.input_size)])
val_tfms.transforms.append(transforms.ToTensor())

print('Benchmarking TResNet-L')

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='TResNet-L',
    paper_arxiv_id='2003.13630',
    input_transform=val_tfms,
    batch_size=256,
    num_workers=args.num_workers,
    num_gpu=1,
    pin_memory=True,
    paper_results={'Top 1 Accuracy': 0.814, 'Top 5 Accuracy': 0.956},
    model_description="Official weights from the author's of the paper."
)

del model
gc.collect()
torch.cuda.empty_cache()

# TResNet-XL
args.model_name = 'tresnet_xl'
model_path = './tresnet_xl.pth'
model = create_model(args)
state = torch.load(model_path, map_location='cpu')['model']
model.load_state_dict(state, strict=True)
model = InplacABN_to_ABN(model)
model = fuse_bn_recursively(model)
model = model.cuda()
model.eval()

val_bs = args.batch_size
val_tfms = transforms.Compose(
    [transforms.Resize(int(args.input_size / args.val_zoom_factor)),
     transforms.CenterCrop(args.input_size)])
val_tfms.transforms.append(transforms.ToTensor())
    
print('Benchmarking TResNet-XL')

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='TResNet-XL',
    paper_arxiv_id='2003.13630',
    input_transform=val_tfms,
    batch_size=128,
    num_workers=args.num_workers,
    num_gpu=1,
    pin_memory=True,
    paper_results={'Top 1 Accuracy': 0.820, 'Top 5 Accuracy': 0.959},
    model_description="Official weights from the author's of the paper."
)
torch.cuda.empty_cache()
