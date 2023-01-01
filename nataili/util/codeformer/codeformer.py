"""
Small modification of CodeFormer from CodeFormer repo.
Uses RealESRGAN x2 from ESRGAN ModelManager.
Allows cache dir to be specified.
Allows device to be specified. NOTE: Device selection seems funky, high CPU usage even when using cuda.
Uses modified FaceRestoreHelper.
"""
import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.registry import ARCH_REGISTRY
from codeformer.facelib.utils.misc import is_gray
from PIL import Image
from realesrgan import RealESRGANer
from torchvision.transforms.functional import normalize

from .face_restoration_helper import FaceRestoreHelper


class CodeFormer(torch.nn.Module):
    def __init__(
        self,
        esrgan_model_manager,
        gfpgan_model_manager,
        codeformer_model_manager,
        upscale=2,
        detection_model="retinaface_resnet50",
        bg_upsampler=None,
        bg_tile=400,
        device="cuda",
    ):
        """
        Args:
            weights (str): path to the pretrained model
            upscale (int): upscale factor
            detection_model (str): Choices: retinaface_resnet50, retinaface_mobile0.25, YOLOv5l, YOLOv5n. Default: retinaface_resnet50
            bg_upsampler (str): Choices: RealESRGAN, None. Default: None
            bg_tile (int): tile size for background upsampling. Default: 400
        """
        super().__init__()
        self.upscale = upscale
        self.detection_model = detection_model
        self.bg_tile = bg_tile

        if bg_upsampler == "realesrgan":
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )
            model_path = esrgan_model_manager.get_model_files("RealESRGAN_x2plus")[0]["path"]
            if "RealESRGAN_x2plus" not in esrgan_model_manager.available_models:
                esrgan_model_manager.download_model("RealESRGAN_x2plus")
            upsampler = RealESRGANer(
                scale=2,
                model_path=model_path,
                model=model,
                tile=bg_tile,
                tile_pad=40,
                pre_pad=0,
                half=True if torch.cuda.is_available() else False,
                device=device,
            )
            self.bg_upsampler = upsampler
        else:
            self.bg_upsampler = None

        self.model = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        )

        model_path = codeformer_model_manager.get_model_files("CodeFormers")[0]["path"]
        model_path = f"{codeformer_model_manager.path}/{model_path}"

        checkpoint = torch.load(model_path, map_location="cpu")["params_ema"]
        self.model.load_state_dict(checkpoint)
        self.model.eval().requires_grad_(False)

        self.face_helper = FaceRestoreHelper(
            self.upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=self.detection_model,
            save_ext="png",
            use_parse=True,
            device=device,
            model_rootpath=gfpgan_model_manager.path,  # GFPGAN uses the same FaceRestoreHelper models
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def to(self, device):
        super().to(device)
        self.face_helper.face_det.to(device)
        self.face_helper.face_parse.to(device)
        self.face_helper.device = device
        return self

    def forward(
        self,
        pil_image: Image.Image,
        fidelity_weight=0.5,
        has_aligned=False,
        only_center_face=False,
        draw_face_bounding_box=False,
    ) -> Image.Image:
        img = np.array(pil_image)

        if has_aligned:
            # the input faces are already cropped and aligned
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            self.face_helper.is_gray = is_gray(img, threshold=5)
            self.face_helper.cropped_faces = [img]
        else:
            self.face_helper.read_image(img)
            # get face landmarks for each face
            self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, resize=640, eye_dist_threshold=5)
            self.face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                with torch.no_grad():
                    output = self.model(cropped_face_t, w=fidelity_weight, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f"\tFailed inference for CodeFormer: {error}")
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype("uint8")
            self.face_helper.add_restored_face(restored_face)

        # paste_back
        if not has_aligned:
            # upsample the background
            if self.bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]
            else:
                bg_img = None
            self.face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = self.face_helper.paste_faces_to_input_image(
                upsample_img=bg_img, draw_box=draw_face_bounding_box
            )

        return Image.fromarray(restored_img)


def test_codeformer():
    model = CodeFormer().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    input_image = Image.open("tests/test.png")
    model(input_image).save("tests/output.png")


def test_codeformer_cpu():
    model = CodeFormer().to(torch.device("cpu"))
    input_image = Image.open("tests/test.png")
    model(input_image).save("tests/output.png")
