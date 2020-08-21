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
from src.models.tresnet.layers.avg_pool import TestTimePoolHead

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

# TResNet-L-2
args.model_name = 'tresnet_l_v2'
model_path = './tresnet_l_2.pth'
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

print('Benchmarking TResNet-L-V2')
# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='TResNet-L-V2',
    paper_arxiv_id='2003.13630',
    input_transform=val_tfms,
    batch_size=280,
    num_workers=args.num_workers,
    num_gpu=1,
    pin_memory=True,
    paper_results={'Top 1 Accuracy': 0.819, 'Top 5 Accuracy': 0.951},
    model_description="TResNet-L-V2."
)

del model
gc.collect()
torch.cuda.empty_cache()

# TResNet-L-V2
args.model_name = 'tresnet_m'
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
    batch_size=432,
    num_workers=args.num_workers,
    num_gpu=1,
    pin_memory=True,
    paper_results={'Top 1 Accuracy': 0.807, 'Top 5 Accuracy': 0.948},
    model_description="Official weights from the author's of the paper."
)

del model
gc.collect()
torch.cuda.empty_cache()

# TResNet-M-288
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
    [transforms.Resize(int(288 / args.val_zoom_factor)),
     transforms.CenterCrop(288)])
val_tfms.transforms.append(transforms.ToTensor())

print('Benchmarking TResNet-M-288')

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='TResNet-M (input=288)',
    paper_arxiv_id='2003.13630',
    input_transform=val_tfms,
    batch_size=352,
    num_workers=args.num_workers,
    num_gpu=1,
    pin_memory=True,
    paper_results={'Top 1 Accuracy': 0.807, 'Top 5 Accuracy': 0.948},
    model_description="Official weights from the author's of the paper."
)

del model
gc.collect()
torch.cuda.empty_cache()

# MTResNet 288-Mean-Max

val_bs = args.batch_size
val_tfms = transforms.Compose(
    [transforms.Resize(288),
     transforms.CenterCrop(288)])
val_tfms.transforms.append(transforms.ToTensor())

model_path = './tresnet_m.pth'
model = create_model(args)
state = torch.load(model_path, map_location='cpu')['model']
model.load_state_dict(state, strict=True)

model = TestTimePoolHead(model)

model = InplacABN_to_ABN(model)
model = fuse_bn_recursively(model)
model = model.cuda()
model.eval()
print('Benchmarking TResNet-M (288-Mean-Max)')

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='TResNet-M (288-Mean-Max)',
    paper_arxiv_id='2003.13630',
    input_transform=val_tfms,
    batch_size=432,
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
    batch_size=250,
    num_workers=args.num_workers,
    num_gpu=1,
    pin_memory=True,
    paper_results={'Top 1 Accuracy': 0.814, 'Top 5 Accuracy': 0.956},
    model_description="Official weights from the author's of the paper."
)

del model
gc.collect()
torch.cuda.empty_cache()

# LTResNet 288-Mean-Max

val_bs = args.batch_size
val_tfms = transforms.Compose(
    [transforms.Resize(288),
     transforms.CenterCrop(288)])
val_tfms.transforms.append(transforms.ToTensor())

model_path = './tresnet_l.pth'
model = create_model(args)
state = torch.load(model_path, map_location='cpu')['model']
model.load_state_dict(state, strict=True)

model = TestTimePoolHead(model)

model = InplacABN_to_ABN(model)
model = fuse_bn_recursively(model)
model = model.cuda()
model.eval()
print('Benchmarking TResNet-L (288-Mean-Max)')

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='TResNet-L (288-Mean-Max)',
    paper_arxiv_id='2003.13630',
    input_transform=val_tfms,
    batch_size=250,
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
    batch_size=250,
    num_workers=args.num_workers,
    num_gpu=1,
    pin_memory=True,
    paper_results={'Top 1 Accuracy': 0.820, 'Top 5 Accuracy': 0.959},
    model_description="Official weights from the author's of the paper."
)

del model
gc.collect()
torch.cuda.empty_cache()

# XLTResNet 288-Mean-Max

val_bs = args.batch_size
val_tfms = transforms.Compose(
    [transforms.Resize(288),
     transforms.CenterCrop(288)])
val_tfms.transforms.append(transforms.ToTensor())

model_path = './tresnet_xl.pth'
model = create_model(args)
state = torch.load(model_path, map_location='cpu')['model']
model.load_state_dict(state, strict=True)

model = TestTimePoolHead(model)

model = InplacABN_to_ABN(model)
model = fuse_bn_recursively(model)
model = model.cuda()
model.eval()
print('Benchmarking TResNet-XL (288-Mean-Max)')

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='TResNet-XL (288-Mean-Max)',
    paper_arxiv_id='2003.13630',
    input_transform=val_tfms,
    batch_size=212,
    num_workers=args.num_workers,
    num_gpu=1,
    pin_memory=True,
    paper_results={'Top 1 Accuracy': 0.820, 'Top 5 Accuracy': 0.959},
    model_description="Official weights from the author's of the paper."
)

del model
gc.collect()
torch.cuda.empty_cache()

# TResNet-M-448
args.model_name = 'tresnet_m'
model_path = './tresnet_m_448.pth'
args.input_size = 448
model = create_model(args)
state = torch.load(model_path, map_location='cpu')['model']
model.load_state_dict(state, strict=True)
model = InplacABN_to_ABN(model)
model = fuse_bn_recursively(model)
model = model.cuda()
model.eval()

val_bs = args.batch_size
val_tfms = transforms.Compose(
    [transforms.Resize((args.input_size, args.input_size))])
val_tfms.transforms.append(transforms.ToTensor())
    
print('Benchmarking TResNet-M 448')

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='TResNet-M (input=448)',
    paper_arxiv_id='2003.13630',
    input_transform=val_tfms,
    batch_size=125,
    num_workers=args.num_workers,
    num_gpu=1,
    pin_memory=True,
    paper_results={'Top 1 Accuracy': 0.832},
    model_description="Official weights from the author's of the paper."
)

del model
gc.collect()
torch.cuda.empty_cache()

# TResNet-L-448
args.model_name = 'tresnet_l'
model_path = './tresnet_l_448.pth'
args.input_size = 448
model = create_model(args)
state = torch.load(model_path, map_location='cpu')['model']
model.load_state_dict(state, strict=True)
model = InplacABN_to_ABN(model)
model = fuse_bn_recursively(model)
model = model.cuda()
model.eval()

val_bs = args.batch_size
val_tfms = transforms.Compose(
    [transforms.Resize((args.input_size, args.input_size))])
val_tfms.transforms.append(transforms.ToTensor())
    
print('Benchmarking TResNet-L 448')

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='TResNet-L (input=448)',
    paper_arxiv_id='2003.13630',
    input_transform=val_tfms,
    batch_size=64,
    num_workers=args.num_workers,
    num_gpu=1,
    pin_memory=True,
    paper_results={'Top 1 Accuracy': 0.838},
    model_description="Official weights from the author's of the paper."
)

del model
gc.collect()
torch.cuda.empty_cache()

# TResNet-XL-448
args.model_name = 'tresnet_xl'
model_path = './tresnet_xl_448.pth'
args.input_size = 448
model = create_model(args)
state = torch.load(model_path, map_location='cpu')['model']
model.load_state_dict(state, strict=True)
model = InplacABN_to_ABN(model)
model = fuse_bn_recursively(model)
model = model.cuda()
model.eval()

val_bs = args.batch_size
val_tfms = transforms.Compose(
    [transforms.Resize((args.input_size, args.input_size))])
val_tfms.transforms.append(transforms.ToTensor())
    
print('Benchmarking TResNet-XL 448')

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='TResNet-XL (input=448)',
    paper_arxiv_id='2003.13630',
    input_transform=val_tfms,
    batch_size=32,
    num_workers=args.num_workers,
    num_gpu=1,
    pin_memory=True,
    paper_results={'Top 1 Accuracy': 0.843},
    model_description="Official weights from the author's of the paper."
)

del model
gc.collect()
torch.cuda.empty_cache()
