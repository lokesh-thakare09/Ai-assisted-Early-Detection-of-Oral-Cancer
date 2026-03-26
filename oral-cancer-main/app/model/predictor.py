
import numpy as np
from pathlib import Path
from typing import Optional
import logging
import os

logger = logging.getLogger(__name__)


class OralCancerPredictor:
    
    
    def __init__(self, config: dict):
        """Initialize the predictor with configuration."""
        self.config = config
        self.model = None
        self.classes = config["model"]["classes"]  # ["Normal", "Oral Cancer"]
        # Hardcoded cancer probability threshold (do not expose in frontend)
        self.confidence_threshold = 0.70
        
        # HuggingFace model info
        self.hf_repo_id = config["model"].get("huggingface_repo_id", "gauravvv7/Oralcancer")
        self.hf_filename = config["model"].get("huggingface_filename", "oral-cancer-model.h5")
        
        # Resolve model path relative to project root
        model_path_str = config["model"]["path"]
        self.model_path = Path(model_path_str)
        
        # If path is relative, make it absolute from project root
        if not self.model_path.is_absolute():
            # Get project root (2 levels up from this file: app/model/predictor.py -> OralCancer/)
            project_root = Path(__file__).parent.parent.parent
            self.model_path = project_root / self.model_path
        
        # Model expects 224x224 input
        self.input_size = (224, 224)
        self.model_type = None
        # Output order config: 'cancer_first' or 'normal_first'
        self.output_order = config["model"].get("output_order", "cancer_first")
        
        # Load the model
        self._load_model()
    
    def _download_from_hf(self) -> Optional[Path]:
        """Download model from Hugging Face."""
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
            
            print("="*80)
            print(f"🔽 DOWNLOADING MODEL FROM HUGGING FACE")
            print(f"   Repository: {self.hf_repo_id}")
            print(f"   Filename: {self.hf_filename}")
            print("="*80)
            
            # Try to detect available .h5 filename in the repo if the configured
            # filename is not present. This makes the downloader robust to
            # repositories that use different names such as `vit_model.h5`.
            try:
                repo_files = list_repo_files(self.hf_repo_id)
                print(f"🔎 Files in repo ({self.hf_repo_id}): {len(repo_files)} items")
                if self.hf_filename not in repo_files:
                    # find first .h5 file
                    h5_candidates = [f for f in repo_files if f.lower().endswith('.h5')]
                    if h5_candidates:
                        old = self.hf_filename
                        self.hf_filename = h5_candidates[0]
                        print(f"⚠️  Configured filename '{old}' not found. Using '{self.hf_filename}' instead.")
            except Exception as e:
                print(f"⚠️  Could not list repo files: {e}")
                # continue and let hf_hub_download attempt the configured filename

            models_dir = self.model_path.parent
            models_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Download directory: {models_dir}")

            downloaded = hf_hub_download(
                repo_id=self.hf_repo_id,
                filename=self.hf_filename,
                local_dir=str(models_dir),
                local_dir_use_symlinks=False
            )
            
            print("="*80)
            print(f"✅ MODEL DOWNLOADED SUCCESSFULLY")
            print(f"   Path: {downloaded}")
            print(f"   Size: {Path(downloaded).stat().st_size / (1024*1024):.2f} MB")
            print("="*80)
            return Path(downloaded)
            
        except Exception as e:
            print("="*80)
            print(f"❌ DOWNLOAD FAILED: {e}")
            print("="*80)
            logger.error(f"Download failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_model(self) -> None:
        """Load the model (Keras or PyTorch)."""
        print("\n" + "="*80)
        print("🚀 INITIALIZING MODEL LOADING")
        print("="*80)
        
        print(f"📍 Looking for model at: {self.model_path}")
        print(f"   File exists: {self.model_path.exists()}")
        
        # Download if needed
        if not self.model_path.exists():
            print("⚠️  Model file not found locally - initiating download...")
            downloaded = self._download_from_hf()
            if downloaded:
                self.model_path = downloaded
            else:
                print("❌ Failed to download model")
        else:
            print(f"✅ Model file found locally ({self.model_path.stat().st_size / (1024*1024):.2f} MB)")
        
        if not self.model_path.exists():
            print("="*80)
            print("❌ MODEL FILE NOT AVAILABLE - Using mock predictions")
            print("="*80)
            logger.error("Model file not found")
            self.model = None
            self.model_type = "mock"
            return
        
        # Detect model type by extension
        model_ext = self.model_path.suffix.lower()
        
        if model_ext in ['.pth', '.pt']:
            self._load_pytorch_model()
        elif model_ext in ['.h5', '.keras']:
            self._load_keras_model()
        else:
            print(f"⚠️  Unknown model format: {model_ext}")
            print("   Attempting to load as Keras model...")
            self._load_keras_model()
    
    def _load_keras_model(self) -> None:
        """Load a Keras/TensorFlow model."""
        try:
            import tensorflow as tf
            
            print("="*80)
            print(f"📥 LOADING KERAS MODEL FROM: {self.model_path}")
            print("="*80)
            
            self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
            self.model_type = "keras"
            
            # Get input size from model
            input_shape = self.model.input_shape
            if input_shape and len(input_shape) == 4:
                h, w = input_shape[1], input_shape[2]
                if h and w:
                    self.input_size = (h, w)
            
            print("="*80)
            print("✅ MODEL LOADED SUCCESSFULLY")
            print(f"   Model type: Keras/TensorFlow")
            print(f"   Input shape: {self.model.input_shape}")
            print(f"   Output shape: {self.model.output_shape}")
            print(f"   Expected input size: {self.input_size}")
            print(f"   Classes (from config): {self.classes}")
            print(f"   Model source: {self.hf_repo_id}")
            print("="*80 + "\n")
            
        except Exception as e:
            print("="*80)
            print(f"❌ KERAS MODEL LOADING FAILED: {e}")
            print("="*80)
            logger.error(f"Failed to load Keras model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.model_type = "mock"
            print("⚠️  Falling back to mock predictions")
    
    def _load_pytorch_model(self) -> None:
        """Load a PyTorch model."""
        try:
            import torch
            
            print("="*80)
            print(f"📥 LOADING PYTORCH MODEL FROM: {self.model_path}")
            print("="*80)
            
            # Load the model checkpoint
            checkpoint = torch.load(str(self.model_path), map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # For now, store the state dict - actual model architecture needs to be defined
            self.model = state_dict
            self.model_type = "pytorch"
            
            print("="*80)
            print("✅ PYTORCH MODEL LOADED SUCCESSFULLY")
            print(f"   Model type: PyTorch")
            print(f"   Expected input size: {self.input_size}")
            print(f"   Classes (from config): {self.classes}")
            print(f"   Checkpoint keys: {list(state_dict.keys())[:5]}...")
            print("="*80 + "\n")
            
        except Exception as e:
            print("="*80)
            print(f"❌ PYTORCH MODEL LOADING FAILED: {e}")
            print("="*80)
            logger.error(f"Failed to load PyTorch model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.model_type = "mock"
            print("⚠️  Falling back to mock predictions")
    
    def predict(self, image: np.ndarray) -> dict:
        """Make a prediction on the input image."""
        print("\n" + "="*80)
        print("🔮 STARTING PREDICTION")
        print(f"   Model type: {self.model_type}")
        print(f"   Input image shape: {image.shape}")
        print("="*80)
        
        if self.model is None:
            print("⚠️  No model loaded - using mock prediction")
            return self._mock_prediction(image)
        
        if self.model_type == "pytorch":
            return self._pytorch_prediction(image)
        elif self.model_type == "keras":
            return self._keras_prediction(image)
        else:
            return self._mock_prediction(image)
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        from PIL import Image as PILImage
        
        # Ensure image is in right format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                img_uint8 = (image * 255).astype(np.uint8)
            else:
                img_uint8 = image.astype(np.uint8)
        else:
            img_uint8 = image
        
        # Convert to PIL and resize to model's expected input
        pil_img = PILImage.fromarray(img_uint8)
        pil_img = pil_img.resize((self.input_size[1], self.input_size[0]), PILImage.Resampling.LANCZOS)
        
        # Convert back to numpy and normalize to [0, 1]
        processed = np.array(pil_img).astype(np.float32) / 255.0
        
        return processed
    
    def _keras_prediction(self, image: np.ndarray) -> dict:
        """Make prediction using Keras model."""
        try:
            print("📊 Preprocessing image...")
            # Preprocess
            processed = self._preprocess(image)
            print(f"   Processed shape: {processed.shape}")
            print(f"   Value range: [{processed.min():.3f}, {processed.max():.3f}]")
            
            # Add batch dimension
            batch_input = np.expand_dims(processed, axis=0)
            print(f"   Batch input shape: {batch_input.shape}")
            
            # Predict
            print("🧠 Running model inference...")
            prediction = self.model.predict(batch_input, verbose=0)
            pred_array = np.array(prediction)

            # Handle both sigmoid (1 output) and softmax (2 outputs) models
            raw_output = None
            if pred_array.shape[-1] == 2:
                # Softmax output with configurable ordering
                a0 = float(pred_array[0][0])
                a1 = float(pred_array[0][1])
                raw_output = pred_array[0].tolist()
                if self.output_order == "cancer_first":
                    cancer_prob = a0
                    normal_prob = a1
                    print(f"📈 Raw model output (softmax, cancer_first): cancer={cancer_prob:.4f}, normal={normal_prob:.4f}")
                else:
                    # normal_first
                    normal_prob = a0
                    cancer_prob = a1
                    print(f"📈 Raw model output (softmax, normal_first): normal={normal_prob:.4f}, cancer={cancer_prob:.4f}")
            else:
                # Sigmoid output: raw_output = P(normal) based on training labels
                raw_output = float(pred_array[0][0])
                print(f"📈 Raw model output (sigmoid): {raw_output:.4f}")

                # Training: class 0=cancer, class 1=normal
                # So raw_output = P(class=1) = P(normal)
                normal_prob = raw_output
                cancer_prob = 1.0 - raw_output

            print(f"   Interpreted probabilities (label mapping):")
            print(f"      • {self.classes[0]}: {normal_prob:.2%}")
            print(f"      • {self.classes[1]}: {cancer_prob:.2%}")

            # Map probabilities to configured class names
            class_probs = {
                self.classes[0]: float(normal_prob),
                self.classes[1]: float(cancer_prob)
            }
            
            # Determine prediction using configured confidence threshold
            try:
                threshold = float(self.confidence_threshold)
            except Exception:
                threshold = 0.6

            if cancer_prob >= threshold:
                predicted_class = self.classes[1]
            else:
                predicted_class = self.classes[0]

            # Report cancer probability as the primary confidence metric
            confidence = cancer_prob
            
            print("="*80)
            print(f"✅ PREDICTION COMPLETE")
            print(f"   Predicted class: {predicted_class}")
            print(f"   Confidence (cancer probability): {confidence:.1%}")
            print(f"   Above threshold ({self.confidence_threshold:.0%}): {confidence >= self.confidence_threshold}")
            print("="*80 + "\n")
            
            return {
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "class_probabilities": class_probs,
                "above_threshold": bool(cancer_prob >= threshold),
                "model_type": "keras",
                "raw_output": raw_output
            }
            
        except Exception as e:
            print("="*80)
            print(f"❌ PREDICTION FAILED: {e}")
            print("="*80)
            logger.error(f"Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            print("⚠️  Falling back to mock prediction")
            return self._mock_prediction(image)
    
    def _pytorch_prediction(self, image: np.ndarray) -> dict:
        """Make prediction using PyTorch model."""
        try:
            import torch
            import torch.nn.functional as F
            
            print("📊 Preprocessing image...")
            # Preprocess
            processed = self._preprocess(image)
            print(f"   Processed shape: {processed.shape}")
            print(f"   Value range: [{processed.min():.3f}, {processed.max():.3f}]")
            
            # Convert to PyTorch tensor and add batch dimension
            # Convert from HWC to CHW format (PyTorch expects channels first)
            if len(processed.shape) == 3:
                processed = np.transpose(processed, (2, 0, 1))
            
            tensor_input = torch.from_numpy(processed).float().unsqueeze(0)
            print(f"   Tensor input shape: {tensor_input.shape}")
            
            # Note: This is a placeholder - actual model architecture needs to be loaded
            # For demonstration, using mock prediction with PyTorch metadata
            print("⚠️  PyTorch model architecture not yet implemented")
            print("   Using mock prediction with PyTorch model loaded...")
            
            # Mock prediction based on image features
            result = self._mock_prediction(image)
            result['model_type'] = 'pytorch'
            result['note'] = 'PyTorch model loaded but architecture needs to be defined'
            
            return result
            
        except Exception as e:
            print("="*80)
            print(f"❌ PYTORCH PREDICTION FAILED: {e}")
            print("="*80)
            logger.error(f"PyTorch prediction failed: {e}")
            import traceback
            traceback.print_exc()
            print("⚠️  Falling back to mock prediction")
            return self._mock_prediction(image)
    
    def _mock_prediction(self, image: np.ndarray) -> dict:
        """Fallback mock prediction."""
        try:
            if len(image.shape) == 3 and image.shape[2] >= 3:
                red = np.mean(image[:,:,0])
                green = np.mean(image[:,:,1])
                total = red + green + 0.001
                red_ratio = red / total
                
                if red_ratio > 0.52:
                    cancer_prob = 0.6 + (red_ratio - 0.52) * 2.0
                else:
                    cancer_prob = 0.3
                
                cancer_prob = max(0.1, min(0.95, cancer_prob))
            else:
                cancer_prob = 0.5
                
        except:
            cancer_prob = 0.5
        
        normal_prob = 1.0 - cancer_prob
        
        class_probs = {
            self.classes[0]: float(normal_prob),
            self.classes[1]: float(cancer_prob)
        }
        
        if cancer_prob >= 0.5:
            predicted_class = self.classes[1]
            confidence = cancer_prob
        else:
            predicted_class = self.classes[0]
            confidence = normal_prob
        
        return {
            "predicted_class": predicted_class,
            "confidence": float(confidence),
            "class_probabilities": class_probs,
            "above_threshold": bool(confidence >= self.confidence_threshold),
            "model_type": "mock",
            "is_mock": True
        }
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_loaded": bool(self.model is not None),
            "model_type": self.model_type,
            "model_path": str(self.model_path),
            "input_size": self.input_size,
            "classes": self.classes,
            "huggingface_repo": self.hf_repo_id,
            "note": "Model output 0=Cancer, 1=Normal (inverted)"
        }
