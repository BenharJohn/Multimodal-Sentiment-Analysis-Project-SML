"""
Data Augmentation Module for Multimodal Sentiment Analysis
Implements image and text augmentation strategies for improved generalization.
"""

import torch
import torch.nn as nn
import random
import re
from PIL import Image, ImageFilter
from torchvision import transforms
from typing import Optional, Dict, List, Tuple, Any


class ImageAugmentation:
    """
    Image augmentation pipeline for training.
    Includes various augmentation strategies to improve model robustness.
    """

    def __init__(
        self,
        image_size: int = 224,
        # Random resized crop
        random_resized_crop: bool = True,
        crop_scale: Tuple[float, float] = (0.8, 1.0),
        # Horizontal flip
        horizontal_flip: bool = True,
        flip_prob: float = 0.5,
        # Color jitter
        color_jitter: bool = True,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        # Rotation
        random_rotation: bool = True,
        rotation_degrees: float = 15,
        # Gaussian blur
        gaussian_blur: bool = True,
        blur_prob: float = 0.1,
        blur_kernel_size: int = 5,
        # Random erasing (cutout)
        random_erasing: bool = False,
        erasing_prob: float = 0.1,
        erasing_scale: Tuple[float, float] = (0.02, 0.1),
        # Normalization (CLIP defaults)
        normalize: bool = True,
        mean: Tuple[float, ...] = (0.48145466, 0.4578275, 0.40821073),
        std: Tuple[float, ...] = (0.26862954, 0.26130258, 0.27577711)
    ):
        """
        Initialize image augmentation pipeline.

        Args:
            image_size: Target image size
            random_resized_crop: Enable random resized crop
            crop_scale: Scale range for random resized crop
            horizontal_flip: Enable horizontal flip
            flip_prob: Probability of horizontal flip
            color_jitter: Enable color jitter
            brightness: Brightness jitter factor
            contrast: Contrast jitter factor
            saturation: Saturation jitter factor
            hue: Hue jitter factor
            random_rotation: Enable random rotation
            rotation_degrees: Maximum rotation degrees
            gaussian_blur: Enable Gaussian blur
            blur_prob: Probability of Gaussian blur
            blur_kernel_size: Kernel size for blur
            random_erasing: Enable random erasing
            erasing_prob: Probability of random erasing
            erasing_scale: Scale range for erasing
            normalize: Apply normalization
            mean: Normalization mean
            std: Normalization std
        """
        self.image_size = image_size
        self.normalize = normalize
        self.mean = mean
        self.std = std

        # Build augmentation pipeline
        augment_list = []

        # Random resized crop or simple resize
        if random_resized_crop:
            augment_list.append(
                transforms.RandomResizedCrop(
                    image_size,
                    scale=crop_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC
                )
            )
        else:
            augment_list.append(
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC
                )
            )

        # Horizontal flip
        if horizontal_flip:
            augment_list.append(transforms.RandomHorizontalFlip(p=flip_prob))

        # Random rotation
        if random_rotation:
            augment_list.append(transforms.RandomRotation(rotation_degrees))

        # Color jitter
        if color_jitter:
            augment_list.append(
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue
                )
            )

        # Gaussian blur (applied probabilistically)
        if gaussian_blur:
            augment_list.append(
                transforms.RandomApply([
                    transforms.GaussianBlur(
                        kernel_size=blur_kernel_size,
                        sigma=(0.1, 2.0)
                    )
                ], p=blur_prob)
            )

        # Convert to tensor
        augment_list.append(transforms.ToTensor())

        # Normalization
        if normalize:
            augment_list.append(transforms.Normalize(mean=mean, std=std))

        # Random erasing (after tensor conversion)
        if random_erasing:
            augment_list.append(
                transforms.RandomErasing(
                    p=erasing_prob,
                    scale=erasing_scale
                )
            )

        self.transform = transforms.Compose(augment_list)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Apply augmentation to image."""
        return self.transform(image)


class TextAugmentation:
    """
    Text augmentation for training.
    Applies various text perturbations to improve model robustness.
    """

    def __init__(
        self,
        # Word dropout
        random_word_dropout: bool = True,
        dropout_prob: float = 0.1,
        # Word swap
        random_word_swap: bool = True,
        swap_prob: float = 0.1,
        # Character perturbation
        random_char_perturbation: bool = False,
        char_perturbation_prob: float = 0.05,
        # Synonym replacement (requires nltk wordnet)
        synonym_replacement: bool = False,
        synonym_prob: float = 0.1
    ):
        """
        Initialize text augmentation.

        Args:
            random_word_dropout: Enable random word dropout
            dropout_prob: Probability of dropping each word
            random_word_swap: Enable random word swap
            swap_prob: Probability of swapping adjacent words
            random_char_perturbation: Enable character perturbation
            char_perturbation_prob: Probability of perturbing characters
            synonym_replacement: Enable synonym replacement
            synonym_prob: Probability of replacing with synonym
        """
        self.random_word_dropout = random_word_dropout
        self.dropout_prob = dropout_prob
        self.random_word_swap = random_word_swap
        self.swap_prob = swap_prob
        self.random_char_perturbation = random_char_perturbation
        self.char_perturbation_prob = char_perturbation_prob
        self.synonym_replacement = synonym_replacement
        self.synonym_prob = synonym_prob

    def __call__(self, text: str) -> str:
        """Apply augmentation to text."""
        if not text or len(text.strip()) == 0:
            return text

        words = text.split()
        if len(words) < 2:
            return text

        # Apply augmentations
        if self.random_word_dropout:
            words = self._word_dropout(words)

        if self.random_word_swap and len(words) >= 2:
            words = self._word_swap(words)

        if self.random_char_perturbation:
            words = self._char_perturbation(words)

        return ' '.join(words)

    def _word_dropout(self, words: List[str]) -> List[str]:
        """Randomly drop words."""
        if len(words) <= 1:
            return words

        new_words = []
        for word in words:
            if random.random() > self.dropout_prob:
                new_words.append(word)

        # Ensure at least one word remains
        if len(new_words) == 0:
            new_words = [random.choice(words)]

        return new_words

    def _word_swap(self, words: List[str]) -> List[str]:
        """Randomly swap adjacent words."""
        words = words.copy()
        for i in range(len(words) - 1):
            if random.random() < self.swap_prob:
                words[i], words[i+1] = words[i+1], words[i]
        return words

    def _char_perturbation(self, words: List[str]) -> List[str]:
        """Apply random character perturbations."""
        keyboard_neighbors = {
            'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'erfcxs',
            'e': 'wsdr', 'f': 'rtgvcd', 'g': 'tyhbvf', 'h': 'yujnbg',
            'i': 'ujko', 'j': 'uikmnh', 'k': 'iolmj', 'l': 'opk',
            'm': 'njk', 'n': 'bhjm', 'o': 'iklp', 'p': 'ol',
            'q': 'wa', 'r': 'edft', 's': 'weadzx', 't': 'rfgy',
            'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc',
            'y': 'tghu', 'z': 'asx'
        }

        new_words = []
        for word in words:
            if random.random() < self.char_perturbation_prob and len(word) > 2:
                # Pick random character position
                idx = random.randint(0, len(word) - 1)
                char = word[idx].lower()
                if char in keyboard_neighbors:
                    new_char = random.choice(keyboard_neighbors[char])
                    if word[idx].isupper():
                        new_char = new_char.upper()
                    word = word[:idx] + new_char + word[idx+1:]
            new_words.append(word)

        return new_words


class MixupAugmentation:
    """
    Mixup augmentation for multimodal data.
    Mixes features and labels from two samples.
    """

    def __init__(self, alpha: float = 0.2):
        """
        Initialize mixup augmentation.

        Args:
            alpha: Beta distribution parameter for mixing coefficient
        """
        self.alpha = alpha

    def __call__(
        self,
        batch: Dict[str, torch.Tensor],
        model: Optional[nn.Module] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, float]:
        """
        Apply mixup to a batch.

        Args:
            batch: Dictionary with input tensors
            model: Optional model for feature-level mixup

        Returns:
            mixed_batch: Batch with mixed inputs
            labels_a: Original labels
            labels_b: Shuffled labels
            lam: Mixing coefficient
        """
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        else:
            lam = 1.0

        batch_size = batch['input_ids'].size(0)

        # Random permutation for mixing
        index = torch.randperm(batch_size)

        # Mix inputs
        mixed_batch = {}
        for key, value in batch.items():
            if key == 'labels':
                continue
            if isinstance(value, torch.Tensor):
                if key == 'pixel_values':
                    # Mix images
                    mixed_batch[key] = lam * value + (1 - lam) * value[index]
                else:
                    # Keep original for text (mixing tokens doesn't work well)
                    mixed_batch[key] = value
            else:
                mixed_batch[key] = value

        labels_a = batch['labels']
        labels_b = batch['labels'][index]

        return mixed_batch, labels_a, labels_b, lam


class CutmixAugmentation:
    """
    Cutmix augmentation for images.
    Cuts a patch from one image and pastes it to another.
    """

    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        """
        Initialize cutmix augmentation.

        Args:
            alpha: Beta distribution parameter
            prob: Probability of applying cutmix
        """
        self.alpha = alpha
        self.prob = prob

    def _rand_bbox(
        self,
        size: Tuple[int, ...],
        lam: float
    ) -> Tuple[int, int, int, int]:
        """Generate random bounding box for cutmix."""
        H, W = size[2], size[3]
        cut_rat = (1 - lam) ** 0.5
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = random.randint(0, W)
        cy = random.randint(0, H)

        bbx1 = max(0, cx - cut_w // 2)
        bby1 = max(0, cy - cut_h // 2)
        bbx2 = min(W, cx + cut_w // 2)
        bby2 = min(H, cy + cut_h // 2)

        return bbx1, bby1, bbx2, bby2

    def __call__(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, float]:
        """
        Apply cutmix to a batch.

        Args:
            batch: Dictionary with input tensors

        Returns:
            mixed_batch: Batch with cutmix applied
            labels_a: Original labels
            labels_b: Mixed labels
            lam: Mixing coefficient (adjusted for actual cut)
        """
        if random.random() > self.prob:
            return batch, batch['labels'], batch['labels'], 1.0

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        batch_size = batch['input_ids'].size(0)
        index = torch.randperm(batch_size)

        mixed_batch = batch.copy()
        pixel_values = batch['pixel_values'].clone()

        # Generate bbox
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(pixel_values.size(), lam)

        # Apply cutmix
        pixel_values[:, :, bby1:bby2, bbx1:bbx2] = \
            pixel_values[index, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda for actual cut
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) /
                   (pixel_values.size()[-1] * pixel_values.size()[-2]))

        mixed_batch['pixel_values'] = pixel_values

        return mixed_batch, batch['labels'], batch['labels'][index], lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """
    Compute loss for mixup augmentation.

    Args:
        criterion: Loss function (e.g., CrossEntropyLoss)
        pred: Model predictions
        labels_a: Original labels
        labels_b: Shuffled labels
        lam: Mixing coefficient

    Returns:
        Mixed loss
    """
    return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)


def build_image_augmentation(config: Dict[str, Any]) -> ImageAugmentation:
    """
    Build image augmentation from config.

    Args:
        config: Augmentation configuration dictionary

    Returns:
        ImageAugmentation instance
    """
    aug_config = config.get('augmentation', {}).get('image', {})

    return ImageAugmentation(
        random_resized_crop=aug_config.get('random_resized_crop', True),
        crop_scale=tuple(aug_config.get('crop_scale', [0.8, 1.0])),
        horizontal_flip=aug_config.get('horizontal_flip', True),
        flip_prob=aug_config.get('flip_prob', 0.5),
        color_jitter=aug_config.get('color_jitter', True),
        brightness=aug_config.get('brightness', 0.2),
        contrast=aug_config.get('contrast', 0.2),
        saturation=aug_config.get('saturation', 0.2),
        hue=aug_config.get('hue', 0.1),
        random_rotation=aug_config.get('random_rotation', True),
        rotation_degrees=aug_config.get('rotation_degrees', 15),
        gaussian_blur=aug_config.get('gaussian_blur', True),
        blur_prob=aug_config.get('blur_prob', 0.1),
        random_erasing=aug_config.get('random_erasing', False),
        erasing_prob=aug_config.get('erasing_prob', 0.1)
    )


def build_text_augmentation(config: Dict[str, Any]) -> TextAugmentation:
    """
    Build text augmentation from config.

    Args:
        config: Augmentation configuration dictionary

    Returns:
        TextAugmentation instance
    """
    aug_config = config.get('augmentation', {}).get('text', {})

    return TextAugmentation(
        random_word_dropout=aug_config.get('random_word_dropout', True),
        dropout_prob=aug_config.get('dropout_prob', 0.1),
        random_word_swap=aug_config.get('random_word_swap', True),
        swap_prob=aug_config.get('swap_prob', 0.1),
        random_char_perturbation=aug_config.get('random_char_perturbation', False),
        char_perturbation_prob=aug_config.get('char_perturbation_prob', 0.05)
    )
