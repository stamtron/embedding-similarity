from __future__ import annotations

import librosa
import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, ClapModel, CLIPModel, CLIPProcessor


class ModalitySimilarity:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Text Model: Sentence Transformers
        # Use a good general purpose model
        try:
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        except Exception as e:
            print(f"Warning: Could not load SentenceTransformer: {e}")
            self.text_model = None

        # Image Model: CLIP
        try:
            self.image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except Exception as e:
            print(f"Warning: Could not load CLIP model: {e}")
            self.image_model = None

        # Audio Model: CLAP
        try:
            self.audio_model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(self.device)
            self.audio_processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
        except Exception as e:
            print(f"Warning: Could not load CLAP model: {e}")
            self.audio_model = None

    def extract_text_embedding(self, text: str | list[str]) -> np.ndarray:
        """Extract embeddings for text using SentenceTransformer."""
        if self.text_model is None:
            raise RuntimeError("Text model not initialized.")

        embeddings = self.text_model.encode(text)
        return embeddings

    def extract_image_embedding(self, image_path: str) -> np.ndarray:
        """Extract embeddings for an image using CLIP."""
        if self.image_model is None:
            raise RuntimeError("Image model not initialized.")

        image = Image.open(image_path).convert("RGB")
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            embeddings = self.image_model.get_image_features(**inputs)

        return embeddings.cpu().numpy().squeeze()

    def extract_audio_embedding(self, audio_path: str) -> np.ndarray:
        """extract embeddings for audio using CLAP."""
        if self.audio_model is None:
            raise RuntimeError("Audio model not initialized.")

        # Load audio at 48k for CLAP (check model requirements, usually 48000 for LAION CLAP)
        y, sr = librosa.load(audio_path, sr=48000)

        # Process audio
        inputs = self.audio_processor(audios=y, sampling_rate=sr, return_tensors="pt").to(self.device)

        with torch.no_grad():
            embeddings = self.audio_model.get_audio_features(**inputs)

        return embeddings.cpu().numpy().squeeze()

    def extract_geo_embedding(self, lat: float, lon: float) -> np.ndarray:
        """
        Extract a simple geospatial embedding using sinusoidal encoding.
        This maps lat/lon to a higher dimensional space to capture proximity even around the dateline/poles.
        """
        # Convert to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)

        # Simple spherical to Cartesian conversion (3D embedding)
        # x = R * cos(lat) * cos(lon)
        # y = R * cos(lat) * sin(lon)
        # z = R * sin(lat)
        # We can use this 3-vector as a simple embedding where euclidean distance approximates geodesic distance

        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)

        return np.array([x, y, z])

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity."""
        # Ensure 1D arrays for single point comparison
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()
        return 1 - cosine(emb1, emb2)

    def compare(self, input1, input2, modality: str) -> dict:
        """Compare two inputs of the same modality."""
        if modality == "text":
            emb1 = self.extract_text_embedding(input1)
            emb2 = self.extract_text_embedding(input2)
        elif modality == "image":
            emb1 = self.extract_image_embedding(input1)
            emb2 = self.extract_image_embedding(input2)
        elif modality == "audio":
            emb1 = self.extract_audio_embedding(input1)
            emb2 = self.extract_audio_embedding(input2)
        elif modality == "geo":
            # Expecting tuples (lat, lon)
            emb1 = self.extract_geo_embedding(*input1)
            emb2 = self.extract_geo_embedding(*input2)
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        similarity = self.compute_similarity(emb1, emb2)
        return {
            "similarity_score": float(similarity),
            "verdict": "Similar" if similarity > 0.8 else "Not Similar"
        }
