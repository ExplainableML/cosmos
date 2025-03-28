import torch
import torch.nn as nn
import sys 
import os
sys.path.append("..")

import open_clip
from open_clip import OPENAI_IMAGENET_TEMPLATES

from mmseg.models.segmentors import BaseSegmentor
from mmseg.models.data_preprocessor import SegDataPreProcessor
from mmengine.structures import PixelData

from mmseg.registry import MODELS

from training.pamr import PAMR

from training.file_utils import pt_load
import copy

import numpy as np
from huggingface_hub import hf_hub_download

def download_weights_from_hf(model_repo, filename):
    # Define the custom cache directory relative to the current script
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    local_path = hf_hub_download(repo_id=model_repo, filename=filename, cache_dir=cache_dir)
    return local_path

def load_model(**model_kwargs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_model_kwargs = {}
    if model_kwargs['siglip']:
        input_model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        input_model_kwargs['init_logit_bias'] = -10
    model, _, _ = open_clip.create_model_and_transforms( 
                                            model_kwargs['model'],
                                            model_kwargs['pretrained'],
                                            precision=model_kwargs['precision'],
                                            device=device,
                                            jit=model_kwargs['torchscript'],
                                            force_quick_gelu=model_kwargs['force_quick_gelu'],
                                            force_custom_text=model_kwargs['force_custom_text'],
                                            force_patch_dropout=model_kwargs['force_patch_dropout'],
                                            force_image_size=model_kwargs['force_image_size'],
                                            image_mean=model_kwargs['image_mean'],
                                            image_std=model_kwargs['image_std'],
                                            image_interpolation=model_kwargs['image_interpolation'],
                                            image_resize_mode=model_kwargs['image_resize_mode'],
                                            use_imagecrop_aug=model_kwargs['use_imagecrop_aug'],
                                            global_crops_number=model_kwargs['global_crops_number'],
                                            local_crops_number=model_kwargs['local_crops_number'],
                                            crop_scale=model_kwargs['crop_scale'],
                                            aug_cfg=model_kwargs['aug_cfg'],
                                            pretrained_image=model_kwargs['pretrained_image'],
                                            output_dict=True,
                                            output_all=model_kwargs['output_all'],
                                            pool_type=model_kwargs['pool_type'],
                                            attentional_pool=model_kwargs['attentional_pool'],
                                            add_zero_attn=model_kwargs['add_zero_attn'],
                                            cosmos=model_kwargs['cosmos'],
                                            **input_model_kwargs,)
    
    ema_model = copy.deepcopy(model)

    if model_kwargs['pretrained']:
        return model

    if model_kwargs['huggingface_model_name'] != '':
        # based on huggingface model name, download the pre-trained weights, the downloaded path is passed as the 'resume' arguments
        huggingface_model_name, huggingface_repo_name = model_kwargs['huggingface_model_name'], model_kwargs['huggingface_repo_name']
        model_kwargs['resume'] = download_weights_from_hf(model_repo=huggingface_repo_name, filename=huggingface_model_name)
        
    checkpoint = pt_load(model_kwargs['resume'], map_location='cpu')

    # resuming a train checkpoint w/ epoch and optimizer state
    start_epoch = checkpoint["epoch"]
    if "state_dict" in checkpoint:
        if model_kwargs['use_ema_model']:
            ema_sd = checkpoint["ema_state_dict"]
            if next(iter(ema_sd.items()))[0].startswith('module'):
                ema_sd = {k[len('module.'):]: v for k, v in ema_sd.items()}
            ema_model.load_state_dict(ema_sd)
            print(f"=> resuming ema checkpoint '{model_kwargs['resume']}' (epoch {start_epoch})")
            return ema_model
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        model.load_state_dict(sd)
        print(f"=> resuming checkpoint '{model_kwargs['resume']}' (epoch {start_epoch})")
        return model
    else:
        """
        sd = checkpoint["student"]
        if next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        model.load_state_dict(sd)
        print(f"=> resuming checkpoint '{model_kwargs['resume']}' (epoch {start_epoch})")
        """
        # Evaluation on Teacher
        ema_sd = checkpoint["teacher"]
        if next(iter(ema_sd.items()))[0].startswith('module'):
            ema_sd = {k[len('module.'):]: v for k, v in ema_sd.items()}
        ema_model.load_state_dict(ema_sd)
        print(f"=> resuming ema checkpoint '{model_kwargs['resume']}' (epoch {start_epoch})")

        return ema_model

@MODELS.register_module()
class CLIPForSegmentation(BaseSegmentor):
    def __init__(self, name_path, device=torch.device('cuda:0'),
                    pamr_steps=0, pamr_stride=(8, 16), prob_thd=0.0, logit_scale=40, 
                    slide_stride=112, slide_crop=224, area_thd=None, **model_kwargs):
        
        data_preprocessor = SegDataPreProcessor(
            mean=[122.771, 116.746, 104.094],
            std=[68.501, 66.632, 70.323],
            rgb_to_bgr=True)
        super().__init__(data_preprocessor=data_preprocessor)

        self.net = load_model(**model_kwargs)
        query_words, self.query_idx = get_cls_idx(name_path)
        self.num_queries = len(query_words)
        self.num_classes = max(self.query_idx) + 1
        self.query_idx = torch.Tensor(self.query_idx).to(torch.int64).to(device)

        query_features = []
        with torch.no_grad():
            for qw in query_words:
                query = open_clip.tokenize([temp(qw) for temp in OPENAI_IMAGENET_TEMPLATES]).to(device) # clip.tokenize([temp(qw) for temp in OPENAI_IMAGENET_TEMPLATES]).to(device)
                feature = self.net.encode_text(query)
                feature = feature['text_features'] if isinstance(feature, dict) else feature
                feature /= feature.norm(dim=-1, keepdim=True)
                feature = feature.mean(dim=0)
                feature /= feature.norm()
                query_features.append(feature.unsqueeze(0))
        self.query_features = torch.cat(query_features, dim=0)
        
        self.dtype = self.query_features.dtype
        self.logit_scale = logit_scale
        self.prob_thd = prob_thd
        self.area_thd = area_thd
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.align_corners = False
        self.use_csa = model_kwargs['use_csa']

        if pamr_steps > 0:
            self.pamr = PAMR(pamr_steps, dilations=pamr_stride).to(device)
        else:
            self.pamr = None

    def forward_feature(self, img, logit_size=None):
        if type(img) == list:
            img = img[0]
        if self.use_csa:
            csa_image_features, _ = self.net.visual(img, return_all=True, csa=True)
            csa_image_features = csa_image_features @ self.net.visual.proj # [B, L-1, C]

            image_features = csa_image_features
            image_features /= image_features.norm(dim=-1, keepdim=True)   
            logits = image_features @ self.query_features.T  
        else:
            image_features, _ = self.net.visual(img, return_all=True, csa=self.use_csa)
            image_features = image_features @ self.net.visual.proj # [B, L-1, C]
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ self.query_features.T

        patch_size = self.net.visual.patch_size
        patch_size = patch_size[0] if isinstance(patch_size, (list, tuple)) else patch_size

        w, h = img[0].shape[-2] // patch_size, img[0].shape[-1] // patch_size
        out_dim = logits.shape[-1]
        logits = logits.permute(0, 2, 1).reshape(-1, out_dim, w, h)

        if logit_size == None:
            logits = nn.functional.interpolate(logits, size=img.shape[-2:], mode='bilinear')
        else:
            logits = nn.functional.interpolate(logits, size=logit_size, mode='bilinear')
        
        return logits

    def forward_slide(self, img, img_metas, stride=112, crop_size=224):
        """Inference by sliding-window with overlap.
        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        if type(img) == list:
            img = img[0].unsqueeze(0)
        if type(stride) == int:
            stride = (stride, stride)
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        out_channels = self.num_queries
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.forward_feature(crop_img)
                preds += nn.functional.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0

        preds = preds / count_mat
        img_size = img_metas[0]['ori_shape'][:2]
        logits = nn.functional.interpolate(preds, size=img_size, mode='bilinear')

        if self.pamr:
            img = nn.functional.interpolate(img, size=img_size, mode='bilinear')
            logits = self.pamr(img, logits.to(img.dtype)).to(self.dtype)

        return logits

    def predict(self, inputs, data_samples):
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]
        
        if self.slide_crop > 0:
            seg_logits = self.forward_slide(inputs, batch_img_metas, self.slide_stride, self.slide_crop)
        else:
            seg_logits = self.forward_feature(inputs, batch_img_metas[0]['ori_shape'])

        return self.postprocess_result(seg_logits, data_samples)
    
    def postprocess_result(self, seg_logits, data_samples):
        batch_size = seg_logits.shape[0]
        for i in range(batch_size):
            seg_logits = seg_logits[i] * self.logit_scale
            seg_logits = seg_logits.softmax(0) # n_queries * w * h

            num_cls, num_queries = max(self.query_idx) + 1, len(self.query_idx)
            if num_cls != num_queries:
                seg_logits = seg_logits.unsqueeze(0)
                cls_index = nn.functional.one_hot(self.query_idx)
                cls_index = cls_index.T.view(num_cls, num_queries, 1, 1)
                seg_logits = (seg_logits * cls_index).max(1)[0]
                seg_pred = seg_logits.argmax(0, keepdim=True)

            if self.area_thd is not None:
                # Force segmentations with area < self.area_thd to 0 (background)
                predictions = nn.functional.one_hot(seg_logits.argmax(0), num_cls).to(seg_logits.dtype)
                area_pred = predictions[:, :, 1:].sum((0, 1), keepdim=True)  # prone background
                area_pred = (area_pred > self.area_thd * area_pred.sum()).to(seg_logits.dtype)          
                seg_logits[1:] *= area_pred.transpose(0, -1)
            
            seg_pred = seg_logits.argmax(0, keepdim=True)
            seg_pred[seg_logits.max(0, keepdim=True)[0] < self.prob_thd] = 0
            
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': seg_pred})
            })

        return data_samples
    
    def _forward(data_samples):
        """
        """
    
    def inference(self, img, batch_img_metas):
        """
        """

    def encode_decode(self, inputs, batch_img_metas):
        """
        """
    
    def extract_feat(self, inputs):
        """
        """
    
    def loss(self, inputs, data_samples):
        """
        """

def get_cls_idx(path):
    with open(path, 'r') as f:
        name_sets = f.readlines()
    num_cls = len(name_sets)

    class_names, class_indices = [], []
    for idx in range(num_cls):
        names_i = name_sets[idx].split(', ')
        class_names += names_i
        class_indices += [idx for _ in range(len(names_i))]
    class_names = [item.replace('\n', '') for item in class_names]
    return class_names, class_indices