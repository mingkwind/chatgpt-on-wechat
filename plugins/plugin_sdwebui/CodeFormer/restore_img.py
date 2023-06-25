import os
import cv2
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from PIL import Image  
import numpy as np
import io

pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

def set_realesrgan():
    if not torch.cuda.is_available():  # CPU
        import warnings
        warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                        'If you really want to use it, please modify the corresponding codes.',
                        category=RuntimeWarning)
        bg_upsampler = None
    else:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.realesrgan_utils import RealESRGANer
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        bg_upsampler = RealESRGANer(
            scale=2,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            model=model,
            tile=400,
            tile_pad=40,
            pre_pad=0,
            half=True)  # need to set False in CPU mode
    return bg_upsampler

def restore_image(params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------ set up background upsampler ------------------
    if params.get('background_enhance', False):
        bg_upsampler = set_realesrgan()
    else:
        bg_upsampler = None

    # ------------------ set up face upsampler ------------------
    if params.get('face_upsample', False):
        if bg_upsampler is not None:
            face_upsampler = bg_upsampler
        else:
            face_upsampler = set_realesrgan()
    else:
        face_upsampler = None

    # ------------------ set up CodeFormer restorer -------------------
    net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                            connect_list=['32', '64', '128', '256']).to(device)
    
    # ckpt_path = 'weights/CodeFormer/codeformer.pth'
    ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'], 
                                    model_dir='weights/CodeFormer', progress=True, file_name=None)
    checkpoint = torch.load(ckpt_path)['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()

    # ------------------ set up FaceRestoreHelper -------------------
    if not params.get('has_aligned', False): 
        print(f'Face detection model: {params.get("detection_model", "retinaface_resnet50")}')
    if bg_upsampler is not None: 
        print(f'Background upsampling: True, Face upsampling: {params.get("face_upsample", False)}')
    else:
        print(f'Background upsampling: False, Face upsampling: {params.get("face_upsample", False)}')

    face_helper = FaceRestoreHelper(
        params.get('upscale', 2),
        face_size=512,
        crop_ratio=(1, 1),
        det_model = params.get('detection_model', 'retinaface_resnet50'),
        save_ext='png',
        use_parse=True,
        device=device)

    # -------------------- start to processing ---------------------
    basename, ext = os.path.splitext(os.path.basename(params['image']))
    result = {}

    img = cv2.imread(params['image'], cv2.IMREAD_COLOR)

    if params.get('has_aligned', False): 
        # the input faces are already cropped and aligned
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        face_helper.cropped_faces = [img]
    else:
        face_helper.read_image(img)
        # get face landmarks for each face
        num_det_faces = face_helper.get_face_landmarks_5(
            only_center_face=params.get('only_center_face', False), resize=640, eye_dist_threshold=5)
        print(f'\tdetect {num_det_faces} faces')
        # align and warp each face
        face_helper.align_warp_face()

    # face restoration for each cropped face
    for idx, cropped_face in enumerate(face_helper.cropped_faces):
        # prepare data
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                output = net(cropped_face_t, w=params.get('codeformer_fidelity', 0.7), adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
        except Exception as error:
            print(f'\tFailed inference for CodeFormer: {error}')
            restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

        restored_face = restored_face.astype('uint8')
        face_helper.add_restored_face(restored_face)

    # paste_back
    if not params.get('has_aligned', False):
        # upsample the background
        if bg_upsampler is not None:
            # Now only support RealESRGAN for upsampling background
            bg_img = bg_upsampler.enhance(img, outscale=params.get('upscale', 2))[0]
        else:
            bg_img = None
        face_helper.get_inverse_affine(None)
        # paste each restored face to the input image
        if params.get('face_upsample', False) and face_upsampler is not None: 
            restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=params.get('draw_box', False), face_upsampler=face_upsampler)
        else:
            restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=params.get('draw_box', False))

    # save restored img
    if not params.get('has_aligned', False) and restored_img is not None:
        data = cv2.imencode('.png', restored_img)[1]
        image_bytes = data.tobytes()
        image_io = io.BytesIO(image_bytes)
        if os.path.exists(params['image']):
            os.remove(params['image'])
        '''
        filename = os.path.basename(params['image'])
        filepath = "/tmp/tmp_"+filename
        imwrite(restored_img, filepath)
        #img = Image.open(filepath)
        #img.save("./2.png")
        img_bytes = io.BytesIO(filepath)
        if os.path.exists(filepath):
            os.remove(filepath)
        if os.path.exists(params['image']):
            os.remove(params['image'])
        '''
        '''
        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(restored_img)
        img.save("./1.png")
        '''

    return image_io

def main():
    params = {
    "codeformer_fidelity": 0.7,
    "upscale": 2,
    "has_aligned": False,
    "only_center_face": False,
    "detection_model": "retinaface_resnet50",
    "draw_box": False,
    "bg_upsampler": "realesrgan",
    "face_upsample": True,
    "background_enhance": True,
    "bg_tile": 400,
    "image": "test/test.png"
    }

    result = restore_image(params)
    print(result)

if __name__=="__main__":
    main()