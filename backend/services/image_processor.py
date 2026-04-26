"""
Image upload and processing service
"""
import os
import io
import hashlib
import logging
from typing import Optional, Dict, Any, Union
from PIL import Image, ImageOps, ImageEnhance
import cv2
import numpy as np
from fastapi import UploadFile
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Service for processing uploaded images"""
    
    def __init__(self):
        self.upload_dir = os.getenv("UPLOAD_DIR", "./uploads")
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", 10485760))  # 10MB
        self.allowed_formats = ['JPEG', 'PNG', 'WEBP', 'BMP']
        self.thumbnail_size = (224, 224)
        self.max_dimensions = (1024, 1024)
        
        # Create upload directory if it doesn't exist
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(os.path.join(self.upload_dir, "thumbnails"), exist_ok=True)
    
    def validate_upload(self, file: UploadFile) -> Dict[str, Any]:
        """Validate uploaded file"""
        try:
            validation_result = {
                'is_valid': False,
                'errors': [],
                'warnings': [],
                'file_info': {}
            }
            
            # Check file size
            if hasattr(file, 'size') and file.size > self.max_file_size:
                validation_result['errors'].append(
                    f"File too large. Maximum size: {self.max_file_size // (1024*1024)}MB"
                )
            
            # Check file type
            if not file.content_type or not file.content_type.startswith('image/'):
                validation_result['errors'].append("File must be an image")
                return validation_result
            
            # Get file info
            validation_result['file_info'] = {
                'filename': file.filename,
                'content_type': file.content_type,
                'size': getattr(file, 'size', 0)
            }
            
            validation_result['is_valid'] = len(validation_result['errors']) == 0
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating upload: {e}")
            return {'is_valid': False, 'errors': [str(e)], 'warnings': [], 'file_info': {}}
    
    def process_upload(self, file: UploadFile) -> Dict[str, Any]:
        """Process uploaded image file"""
        try:
            # Validate file
            validation = self.validate_upload(file)
            if not validation['is_valid']:
                return validation
            
            # Read file content
            file_content = file.file.read()
            
            # Reset file pointer
            file.file.seek(0)
            
            # Open image
            image = Image.open(io.BytesIO(file_content))
            
            # Process image
            processed_result = self.process_image(image, file.filename)
            
            # Add file info
            processed_result['file_info'] = validation['file_info']
            processed_result['is_valid'] = True
            
            return processed_result
            
        except Exception as e:
            logger.error(f"Error processing upload: {e}")
            return {'is_valid': False, 'errors': [str(e)], 'warnings': [], 'file_info': {}}
    
    def process_image(self, image: Image.Image, filename: Optional[str] = None) -> Dict[str, Any]:
        """Process PIL Image"""
        try:
            result = {
                'is_valid': False,
                'errors': [],
                'warnings': [],
                'processed_images': {},
                'metadata': {}
            }
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                result['warnings'].append(f"Converted from {image.mode} to RGB")
            
            # Auto-orient image
            image = ImageOps.exif_transpose(image)
            
            # Get original metadata
            result['metadata'] = {
                'original_size': image.size,
                'original_mode': image.mode,
                'format': image.format or 'JPEG'
            }
            
            # Generate unique filename
            if filename:
                name, ext = os.path.splitext(filename)
                unique_name = f"{name}_{hash(filename) % 10000}"
            else:
                unique_name = f"image_{hash(image.tobytes()) % 10000}"
            
            # Process different sizes
            processed_images = {}
            
            # Original (resized if too large)
            original_image = self.resize_if_needed(image)
            original_path = os.path.join(self.upload_dir, f"{unique_name}.jpg")
            original_image.save(original_path, 'JPEG', quality=85, optimize=True)
            processed_images['original'] = {
                'path': original_path,
                'size': original_image.size,
                'url': f"/uploads/{os.path.basename(original_path)}"
            }
            
            # Thumbnail
            thumbnail = self.create_thumbnail(original_image)
            thumbnail_path = os.path.join(self.upload_dir, "thumbnails", f"{unique_name}_thumb.jpg")
            thumbnail.save(thumbnail_path, 'JPEG', quality=80, optimize=True)
            processed_images['thumbnail'] = {
                'path': thumbnail_path,
                'size': thumbnail.size,
                'url': f"/uploads/thumbnails/{os.path.basename(thumbnail_path)}"
            }
            
            # Medium size
            medium = self.create_medium_size(original_image)
            medium_path = os.path.join(self.upload_dir, f"{unique_name}_medium.jpg")
            medium.save(medium_path, 'JPEG', quality=85, optimize=True)
            processed_images['medium'] = {
                'path': medium_path,
                'size': medium.size,
                'url': f"/uploads/{os.path.basename(medium_path)}"
            }
            
            result['processed_images'] = processed_images
            result['is_valid'] = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {'is_valid': False, 'errors': [str(e)], 'warnings': [], 'metadata': {}}
    
    def resize_if_needed(self, image: Image.Image) -> Image.Image:
        """Resize image if it exceeds maximum dimensions"""
        try:
            width, height = image.size
            max_width, max_height = self.max_dimensions
            
            if width <= max_width and height <= max_height:
                return image
            
            # Calculate new size maintaining aspect ratio
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return image
    
    def create_thumbnail(self, image: Image.Image) -> Image.Image:
        """Create thumbnail from image"""
        try:
            # Create thumbnail maintaining aspect ratio
            image.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
            
            # Create square thumbnail with padding if needed
            width, height = image.size
            thumb_width, thumb_height = self.thumbnail_size
            
            if width == thumb_width and height == thumb_height:
                return image
            
            # Create new image with white background
            thumbnail = Image.new('RGB', self.thumbnail_size, (255, 255, 255))
            
            # Calculate position to center the image
            x = (thumb_width - width) // 2
            y = (thumb_height - height) // 2
            
            # Paste the image
            thumbnail.paste(image, (x, y))
            
            return thumbnail
            
        except Exception as e:
            logger.error(f"Error creating thumbnail: {e}")
            return image.resize(self.thumbnail_size, Image.Resampling.LANCZOS)
    
    def create_medium_size(self, image: Image.Image) -> Image.Image:
        """Create medium-sized image"""
        try:
            width, height = image.size
            medium_size = (512, 512)
            
            if width <= medium_size[0] and height <= medium_size[1]:
                return image
            
            # Calculate new size maintaining aspect ratio
            ratio = min(medium_size[0] / width, medium_size[1] / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
        except Exception as e:
            logger.error(f"Error creating medium size: {e}")
            return image
    
    def enhance_image(self, image: Image.Image, enhancement_type: str = 'auto') -> Image.Image:
        """Enhance image quality"""
        try:
            if enhancement_type == 'auto':
                # Auto-enhance
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.1)
                
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.05)
                
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(1.05)
            
            elif enhancement_type == 'contrast':
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.2)
            
            elif enhancement_type == 'brightness':
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(1.1)
            
            elif enhancement_type == 'sharpness':
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.2)
            
            return image
            
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return image
    
    def extract_features(self, image: Image.Image) -> Dict[str, Any]:
        """Extract features from image using OpenCV"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            features = {}
            
            # Basic statistics
            features.update({
                'width': image.width,
                'height': image.height,
                'aspect_ratio': image.width / image.height,
                'total_pixels': image.width * image.height
            })
            
            # Color analysis
            if len(img_array.shape) == 3:  # RGB image
                # Average color
                avg_color = np.mean(img_array, axis=(0, 1))
                features['average_color'] = avg_color.tolist()
                
                # Color distribution (simplified histogram)
                hist_r = np.histogram(img_array[:, :, 0], bins=16, range=(0, 256))[0]
                hist_g = np.histogram(img_array[:, :, 1], bins=16, range=(0, 256))[0]
                hist_b = np.histogram(img_array[:, :, 2], bins=16, range=(0, 256))[0]
                
                features['color_histogram'] = {
                    'red': hist_r.tolist(),
                    'green': hist_g.tolist(),
                    'blue': hist_b.tolist()
                }
                
                # Dominant color channel
                color_sums = [np.sum(hist_r), np.sum(hist_g), np.sum(hist_b)]
                dominant_channel = ['red', 'green', 'blue'][np.argmax(color_sums)]
                features['dominant_channel'] = dominant_channel
            
            # Edge detection (simplified)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            features['edge_density'] = float(edge_density)
            
            # Texture (simplified - standard deviation)
            texture = np.std(gray)
            features['texture_complexity'] = float(texture)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}
    
    def cleanup_old_files(self, max_age_hours: int = 24):
        """Clean up old uploaded files"""
        try:
            import time
            
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            for root, dirs, files in os.walk(self.upload_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_age = current_time - os.path.getctime(file_path)
                    
                    if file_age > max_age_seconds:
                        try:
                            os.remove(file_path)
                            logger.info(f"Removed old file: {file_path}")
                        except Exception as e:
                            logger.error(f"Error removing file {file_path}: {e}")
            
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")

# Global instance
image_processor = ImageProcessor()

def get_image_processor() -> ImageProcessor:
    """Get global image processor instance"""
    return image_processor
