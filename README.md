# Embedding Similarity Showcase

A hands-on project to explore and assist understand Embeddings, Cosine Similarity, and how different models and modalities (Text, Image, Audio, Geospatial) work.

## 🚀 Overview

This repository demonstrates how to generate embeddings and calculate similarity for various data types using state-of-the-art models.

### Key Concepts
- **Embeddings**: Vector representations of data where similar items are closer in the multi-dimensional space.
- **Cosine Similarity**: A metric used to measure how similar the documents are irrespective of their size. It measures the cosine of the angle between two vectors projected in a multi-dimensional space.

## 📦 Features

- **Text Embeddings**: Using `sentence-transformers` (e.g., `all-MiniLM-L6-v2`) to semantic search.
- **Image Embeddings**: Using OpenAI's `CLIP` to find similar images or image-text pairs.
- **Audio Embeddings**: Using `CLAP` to map audio to vector space.
- **Geospatial Embeddings**: Custom sinusoidal encoding for coordinate similarity.

## 📂 Project Structure

```
.
├── notebooks/                  # Interactive notebooks for each modality
│   ├── 01_text_embeddings.ipynb
│   ├── 02_image_embeddings.ipynb
│   ├── 03_audio_embeddings.ipynb
│   └── 04_geo_embeddings.ipynb
├── src/
│   └── similarity_finder/      # Core logic for embedding extraction
├── tests/                      # Unit tests
├── pyproject.toml              # Project configuration
└── README.md
```

## 🛠️ Usage

    ```bash
    uv sync
    ```

2.  **Run Notebooks**:
    Navigate to the `notebooks/` directory and run `jupyter notebook` to explore the examples.

## 🧠 Models Used

| Modality | Model / Method | Description |
|---|---|---|
| **Text** | `all-MiniLM-L6-v2` | Fast and effective sentence embeddings. |
| **Image**| `CLIP` (ViT-B/32) | Contrastive Language-Image Pre-Training. |
| **Audio**| `CLAP` (LAION) | Contrastive Language-Audio Pre-training. |
| **Geo**  | Sinusoidal Encoding | Maps Lat/Lon to 3D continuous space. |

## 🤝 Contributing

Feel free to open issues or submit PRs if you want to add more modalities (e.g., Video, Graph) or better models!
