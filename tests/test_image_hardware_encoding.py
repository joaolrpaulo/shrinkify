"""Tests for opt-in GPU-backed AVIF command selection."""

import unittest
from pathlib import Path
from unittest.mock import patch

from core.converter import _image_cmd
from core.scanner import MediaFile


class ImageHardwareEncodingTests(unittest.TestCase):
    def setUp(self):
        self.media_file = MediaFile(Path('source.jpg'), 100, 'image')
        self.output_path = Path('result.avif')

    @patch('core.converter.get_hw_image_encoder', return_value='av1_nvenc')
    def test_gpu_mode_uses_available_av1_encoder(self, mock_encoder):
        cmd, is_hardware = _image_cmd(
            self.media_file, self.output_path, quality=75, use_hw_accel=True
        )

        self.assertTrue(is_hardware)
        self.assertIn('av1_nvenc', cmd)
        self.assertNotIn('libaom-av1', cmd)
        mock_encoder.assert_called_once()

    @patch('core.converter.get_hw_image_encoder')
    def test_cpu_mode_does_not_probe_or_use_gpu(self, mock_encoder):
        cmd, is_hardware = _image_cmd(
            self.media_file, self.output_path, quality=75, use_hw_accel=False
        )

        self.assertFalse(is_hardware)
        self.assertIn('libaom-av1', cmd)
        mock_encoder.assert_not_called()

    @patch('core.converter.get_hw_image_encoder', return_value='av1_vaapi')
    def test_vaapi_mode_initializes_and_uploads_to_the_vaapi_device(self, mock_encoder):
        cmd, is_hardware = _image_cmd(
            self.media_file, self.output_path, quality=75, use_hw_accel=True
        )

        self.assertTrue(is_hardware)
        self.assertIn('av1_vaapi', cmd)
        self.assertIn('vaapi=va', cmd)
        self.assertIn('format=nv12,hwupload', cmd)

    @patch('core.converter.get_hw_image_encoder', return_value='av1_vulkan')
    def test_vulkan_mode_initializes_and_uploads_to_the_vulkan_device(self, mock_encoder):
        cmd, is_hardware = _image_cmd(
            self.media_file, self.output_path, quality=75, use_hw_accel=True
        )

        self.assertTrue(is_hardware)
        self.assertIn('av1_vulkan', cmd)
        self.assertIn('vulkan=vk', cmd)
        self.assertIn('format=yuv420p,hwupload', cmd)


if __name__ == '__main__':
    unittest.main()
