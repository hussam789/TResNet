import gc

from torchbench.image_classification import ImageNet
import urllib.request
import torch
from torchvision.transforms import transforms
from src.helper_functions.helper_functions import validate, create_dataloader, create_val_tfm, \
    upload_data_to_gpu
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
val_tfms = create_val_tfm(args)

#### TResNet-M ####
args.model_name = 'tresnet_m'
model_path = './tresnet_m.pth'
model = create_model(args)
state = torch.load(model_path, map_location='cpu')['model']
model.load_state_dict(state, strict=True)
model = InplacABN_to_ABN(model)
model = fuse_bn_recursively(model)
model = model.cuda()
model = model.half()
model.eval()

# Run the benchmark
print('Benchmarking TResNet-M')
for i in range(2):  # Two times for caching
    ImageNet.benchmark(
        model=model,
        paper_model_name='TResNet-M-FP16',
        paper_arxiv_id='2003.13630',
        input_transform=val_tfms,
        batch_size=800,
        num_workers=args.num_workers,
        num_gpu=1,
        pin_memory=True,
        paper_results={'Top 1 Accuracy': 0.807, 'Top 5 Accuracy': 0.948},
        model_description="Official weights from the author's of the paper.",
        send_data_to_device=upload_data_to_gpu
    )

del model
gc.collect()
torch.cuda.empty_cache()

#### TResNet-L-2 ####
args.model_name = 'tresnet_l_v2'
model_path = './tresnet_l_2.pth'
model = create_model(args)
state = torch.load(model_path, map_location='cpu')['model']
model.load_state_dict(state, strict=True)
model = InplacABN_to_ABN(model)
model = fuse_bn_recursively(model)
model = model.cuda()
model = model.half()
model.eval()

# Run the benchmark
print('Benchmarking TResNet-L-V2-FP16 ')
for i in range(2):  # Two times for caching
    ImageNet.benchmark(
        model=model,
        paper_model_name='TResNet-L-V2-FP16',
        paper_arxiv_id='2003.13630',
        input_transform=val_tfms,
        batch_size=600,
        num_workers=args.num_workers,
        num_gpu=1,
        pin_memory=True,
        paper_results={'Top 1 Accuracy': 0.819, 'Top 5 Accuracy': 0.951},
        model_description="TResNet-L-V2.",
        send_data_to_device=upload_data_to_gpu,
    )

del model
gc.collect()
torch.cuda.empty_cache()
