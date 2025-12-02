import torch
import torch.nn.functional as F
from typing import Union, List, Optional, Tuple
import numpy as np
from PIL import Image


class CLIPImageProcessor:
    def __init__(
        self,
        size: Union[int, Tuple[int, int]] = 336,
        crop_size: Optional[Union[int, Tuple[int, int]]] = None,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        do_normalize: bool = True,
        do_resize: bool = True,
        do_center_crop: bool = False,
        do_convert_rgb: bool = True,
        interpolation: str = "bicubic",
    ):
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        
        if crop_size is not None and isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size or size
        self.mean = mean or [0.48145466, 0.4578275, 0.40821073]
        self.std = std or [0.26862954, 0.26130258, 0.27577711]
        
        self.do_normalize = do_normalize
        self.do_resize = do_resize
        self.do_center_crop = do_center_crop
        self.do_convert_rgb = do_convert_rgb
        
        # Map interpolation string to torch mode
        self.interpolation_map = {
            "nearest": "nearest",
            "bilinear": "bilinear",
            "bicubic": "bicubic",
            "lanczos": "bicubic",  # Approximate lanczos with bicubic
        }
        self.interpolation = self.interpolation_map.get(interpolation, "bicubic")
    
    def resize(
        self,
        image: torch.Tensor,
        size: Tuple[int, int],
        interpolation: str = "bicubic",
    ) -> torch.Tensor:
        """
        Resize image tensor
        
        Args:
            image: Image tensor of shape (C, H, W) or (B, C, H, W)
            size: Target size (height, width)
            interpolation: Interpolation method
        
        Returns:
            Resized image tensor
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_after = True
        else:
            squeeze_after = False
        
        # Use F.interpolate for efficient resizing
        resized = F.interpolate(
            image,
            size=size,
            mode=interpolation,
            align_corners=False if interpolation == "bilinear" else None,
            antialias=True if interpolation in ["bilinear", "bicubic"] else False,
        )
        
        if squeeze_after:
            resized = resized.squeeze(0)
        
        return resized
    
    def center_crop(
        self,
        image: torch.Tensor,
        crop_size: Tuple[int, int],
    ) -> torch.Tensor:
        if image.dim() == 3:
            _, height, width = image.shape
        else:
            _, _, height, width = image.shape
        
        crop_height, crop_width = crop_size
        
        if height == crop_height and width == crop_width:
            return image
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        
        if image.dim() == 3:
            return image[:, top:top+crop_height, left:left+crop_width]
        else:
            return image[:, :, top:top+crop_height, left:left+crop_width]
    
    def normalize(
        self,
        image: torch.Tensor,
        mean: List[float],
        std: List[float],
    ) -> torch.Tensor:
        mean_tensor = torch.tensor(mean, dtype=image.dtype, device=image.device)
        std_tensor = torch.tensor(std, dtype=image.dtype, device=image.device)
        
        if image.dim() == 3:
            mean_tensor = mean_tensor.view(3, 1, 1)
            std_tensor = std_tensor.view(3, 1, 1)
        else:
            mean_tensor = mean_tensor.view(1, 3, 1, 1)
            std_tensor = std_tensor.view(1, 3, 1, 1)
        
        return (image - mean_tensor) / std_tensor
    
    def pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """
        Convert PIL image to tensor
        
        Args:
            image: PIL Image
        
        Returns:
            Image tensor of shape (C, H, W) with values in [0, 1]
        """
        # Convert to RGB if needed
        if self.do_convert_rgb and image.mode != "RGB":
            image = image.convert("RGB")
        
        # Convert to numpy array
        np_image = np.array(image, dtype=np.float32)
        
        # Normalize to [0, 1]
        np_image = np_image / 255.0
        
        # Convert to tensor and change from HWC to CHW
        tensor_image = torch.from_numpy(np_image)
        if tensor_image.dim() == 3:
            tensor_image = tensor_image.permute(2, 0, 1)
        
        return tensor_image
    
    def process_single_image(
        self,
        image: Union[Image.Image, torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """
        Process a single image
        
        Args:
            image: Input image (PIL, tensor, or numpy array)
        
        Returns:
            Processed image tensor of shape (C, H, W)
        """
        # Convert to tensor if needed
        if isinstance(image, Image.Image):
            image_tensor = self.pil_to_tensor(image)
        elif isinstance(image, np.ndarray):
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            if image.max() > 1:
                image = image / 255.0
            image_tensor = torch.from_numpy(image)
            if image_tensor.dim() == 3 and image_tensor.shape[-1] == 3:
                image_tensor = image_tensor.permute(2, 0, 1)
        else:
            image_tensor = image
        
        # Ensure 3 channels
        if image_tensor.dim() == 2:
            image_tensor = image_tensor.unsqueeze(0)
        
        # Resize if needed
        if self.do_resize:
            image_tensor = self.resize(image_tensor, self.size, self.interpolation)
        
        # Center crop if needed
        if self.do_center_crop:
            image_tensor = self.center_crop(image_tensor, self.crop_size)
        
        # Normalize if needed
        if self.do_normalize:
            image_tensor = self.normalize(image_tensor, self.mean, self.std)
        
        return image_tensor
    
    def __call__(
        self,
        images: Union[
            Image.Image,
            torch.Tensor,
            np.ndarray,
            List[Union[Image.Image, torch.Tensor, np.ndarray]]
        ],
        return_tensors: str = "pt",
    ) -> dict:
        """
        Process images for CLIP model
        
        Args:
            images: Single image or list of images
            return_tensors: Format of output tensors ("pt" for PyTorch)
        
        Returns:
            Dictionary with "pixel_values" key containing processed images
        """
        # Handle single image
        if not isinstance(images, list):
            images = [images]
        
        # Process all images
        processed_images = []
        for image in images:
            processed = self.process_single_image(image)
            processed_images.append(processed)
        
        # Stack into batch
        pixel_values = torch.stack(processed_images, dim=0)
        
        return {"pixel_values": pixel_values}
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """
        Load processor configuration from pretrained model
        
        Args:
            model_name_or_path: Model name or path
            **kwargs: Additional arguments
        
        Returns:
            Configured CLIPImageProcessor
        """
        # Default CLIP configuration
        # Can be extended to load from config files
        return cls(**kwargs)


def create_clip_image_processor(
    size: int = 336,
    do_normalize: bool = True,
    do_resize: bool = True,
) -> CLIPImageProcessor:
    """
    Create a standard CLIP image processor
    
    Args:
        size: Image size
        do_normalize: Whether to normalize
        do_resize: Whether to resize
    
    Returns:
        CLIPImageProcessor instance
    """
    return CLIPImageProcessor(
        size=size,
        do_normalize=do_normalize,
        do_resize=do_resize,
    )