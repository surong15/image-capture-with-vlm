# VLM + Milvus Vector Database Integration System

A desktop application that integrates a Vision Language Model (VLM) with a Milvus vector database, supporting image-based Q&A, conversation history management, vector search, and more.

## Key Features

### Core Functions
- **Real-time Image Q&A**: Capture photos through a camera and perform image understanding and question answering using an AI model.
- **Multi-model Support**: Select from locally installed Ollama models that support image input.
- **Vector Database Storage**: Store conversations and images directly in the Milvus vector database.
- **Conversation History Management**: View and search through previously stored conversations.
- **Image Viewer**: Inspect past images stored in conversation history.

### Technical Highlights
- **Embedded Image Storage**: Images are stored directly in the vector database as base64-encoded strings.
- **Semantic Search**: Perform vector similarity search to retrieve related conversations.
- **Multimodal Support**: Works with multimodal models like LLaVA, Gemma3, and others.
- **Real-time Processing**: Live camera preview and fast image analysis.

## System Requirements

### Hardware
- Camera device (built-in or external)
- At least 8GB RAM (16GB recommended)
- GPU with CUDA support (optional for faster inference)

### Software
- Python 3.8+
- Docker (for running Milvus)
- Ollama (local AI model server)

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and Start Ollama

```bash
# Install Ollama (macOS)
curl -fsSL https://ollama.ai/install.sh | sh

# Start the Ollama service
ollama serve

# Download an image-capable model
ollama pull llava-phi3:latest
# or
ollama pull llava:latest
```

### 3. Launch the Milvus Vector Database

```bash
docker-compose up -d
```

### 4. (Optional) Launch Attu Management Interface

```bash
docker run -p 8000:3000 -e MILVUS_URL=localhost:19530 zilliz/attu:latest
```

## Project Structure

```
vlm-project/
├── vlm_with_milvus_model_select.py  # Main application (model-select version)
├── requirements.txt                  # Python dependencies
├── docker-compose.yml               # Milvus container configuration
├── README.md                        # Project documentation
└── volumes/                         # Milvus data storage directory
```

## Usage

### Start the Application

```bash
python vlm_with_milvus_model_select.py
```

### Basic Workflow

1. **Start Camera**: Click “Start Camera”
2. **Select Model**: Choose an AI model from the dropdown
3. **Adjust Parameters**: Configure temperature and other options
4. **Enter Question**: Type your question in the input box
5. **Capture Image for Analysis**: Click “Capture & Analyze”
6. **View Results**: The AI response will appear in the right panel

### Advanced Features

#### Conversation History Management
- **Search Conversations**: Enter keywords and press Enter or click “Search”
- **View History**: Click “View History” to browse saved conversations
- **View Image**: Enter a conversation ID to display the stored image

#### Model Management
- The system automatically detects installed Ollama models.
- Supports all models capable of image input (e.g. LLaVA, Phi3 variants).
- Models can be switched instantly within the GUI.

## Configuration Details

### Model Configuration
- **Default Model**: `llava-phi3:latest`
- **Supported Models**: Any Ollama model supporting vision input
- **Runtime Switching**: Models can be swapped dynamically through the UI

### Database Configuration
- **Vector Database**: Milvus
- **Collection Name**: `vlm_conversations_with_images`
- **Vector Dimension**: 384 (using `paraphrase-multilingual-MiniLM-L12-v2`)
- **Index Type**: `IVF_FLAT` with cosine similarity

### Image Handling
- **Storage Format**: JPEG (base64 encoded)
- **Max Size**: 800px (auto resized)
- **Quality Control**: Automatically adjusted to meet DB size constraints

