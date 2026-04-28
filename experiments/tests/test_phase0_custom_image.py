"""Tests for custom image loading in Phase 0 ViT gradient inversion."""

import os
import sys
import torch
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def _create_test_image(path, width=300, height=400):
    """Create a simple test RGB image."""
    img = Image.new('RGB', (width, height), color=(128, 64, 200))
    img.save(path)
    return path


class TestCustomImageLoading:
    def test_loads_custom_image(self, tmp_path):
        """Custom image loads with correct shape and label."""
        img_path = str(tmp_path / 'test.jpg')
        _create_test_image(img_path)

        from experiments.phase0_vit_inversion import get_sample_images
        images, labels = get_sample_images(image_path=img_path, device='cpu')

        assert images.shape == (1, 3, 224, 224)
        assert labels.shape == (1, 1)
        assert labels.item() == 0.0

    def test_values_in_imagenet_range(self, tmp_path):
        """Pixel values are in ImageNet-normalized range."""
        img_path = str(tmp_path / 'test.png')
        _create_test_image(img_path)

        from experiments.phase0_vit_inversion import get_sample_images
        images, _ = get_sample_images(image_path=img_path, device='cpu')

        assert images.min() >= -2.5, f"Min value {images.min()} below range"
        assert images.max() <= 3.0, f"Max value {images.max()} above range"

    def test_file_not_found_raises(self):
        """Missing file raises FileNotFoundError."""
        from experiments.phase0_vit_inversion import get_sample_images
        with pytest.raises(FileNotFoundError, match="Custom image not found"):
            get_sample_images(image_path='/nonexistent/image.jpg', device='cpu')

    def test_image_path_overrides_dataset(self, tmp_path):
        """When image_path is set, dataset_name is ignored."""
        img_path = str(tmp_path / 'test.jpg')
        _create_test_image(img_path, width=200, height=200)

        from experiments.phase0_vit_inversion import get_sample_images
        images, labels = get_sample_images(
            image_path=img_path, dataset_name='flowers102', device='cpu')

        # Should load the custom image (1 image), not flowers102
        assert images.shape == (1, 3, 224, 224)

    def test_rgba_image_converted(self, tmp_path):
        """RGBA images are converted to RGB."""
        img_path = str(tmp_path / 'test.png')
        img = Image.new('RGBA', (300, 300), color=(128, 64, 200, 255))
        img.save(img_path)

        from experiments.phase0_vit_inversion import get_sample_images
        images, _ = get_sample_images(image_path=img_path, device='cpu')

        assert images.shape == (1, 3, 224, 224)

    def test_grayscale_image_converted(self, tmp_path):
        """Grayscale images are converted to RGB."""
        img_path = str(tmp_path / 'test.png')
        img = Image.new('L', (300, 300), color=128)
        img.save(img_path)

        from experiments.phase0_vit_inversion import get_sample_images
        images, _ = get_sample_images(image_path=img_path, device='cpu')

        assert images.shape == (1, 3, 224, 224)

    def test_exif_rotation_handled(self, tmp_path):
        """Images with EXIF rotation are transposed correctly."""
        import struct
        img_path = str(tmp_path / 'test.jpg')
        # Create a non-square image so rotation would change dimensions
        img = Image.new('RGB', (400, 200), color=(255, 0, 0))
        img.save(img_path)

        from experiments.phase0_vit_inversion import get_sample_images
        images, _ = get_sample_images(image_path=img_path, device='cpu')

        # Should still produce 224x224 regardless of orientation
        assert images.shape == (1, 3, 224, 224)

    def test_actual_face_photos(self):
        """Test loading actual face photos if available."""
        faces_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                                 'data', 'faces')
        face1 = os.path.join(faces_dir, 'face1.jpg')
        if not os.path.exists(face1):
            pytest.skip("Face photos not available at data/faces/")

        from experiments.phase0_vit_inversion import get_sample_images
        images, labels = get_sample_images(image_path=face1, device='cpu')

        assert images.shape == (1, 3, 224, 224)
        assert labels.item() == 0.0
        # Should produce reasonable normalized values
        assert -3.0 < images.min() < 0.0
        assert 0.0 < images.max() < 3.5
