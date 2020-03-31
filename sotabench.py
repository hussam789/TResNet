from torchbench.image_classification import ImageNet
import urllib.request
import torch
from torchvision.transforms import transforms
from src.helper_functions.helper_functions import validate, create_dataloader
from src.models import create_model
import argparse


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
model_path, _ = urllib.request.urlretrieve('https://drive.google.com/open?id=12_VnXYI-4JaUYOOIsZXYCJdpiLCJ4dHV', 'mtresnet.pth')
print(model_path)
model = create_model(args).cuda()
state = torch.load(model_path, map_location='cpu')['model']
model.load_state_dict(state, strict=True)
model.eval()

val_bs = args.batch_size
val_tfms = transforms.Compose(
    [transforms.Resize(int(args.input_size / args.val_zoom_factor)),
     transforms.CenterCrop(args.input_size)])
val_tfms.transforms.append(transforms.ToTensor())
    
# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='TResNet-M',
    paper_arxiv_id='2003.13630',
    input_transform=val_tfms,
    batch_size=256,
    num_gpu=1,
    model_description="Official weights from the author's of the paper."
)
torch.cuda.empty_cache()
