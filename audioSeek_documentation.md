# AudioSeek Data Pipeline Report

**Platform:** Docker, Docker Compose
**Orchestration:** Apache Airflow
**Pipeline Objective:** Automated audiobook processing for transcription, chunking, embedding, and question answering to create a searchable knowledge base. This enables users to quickly find information within audiobooks, get summaries of chapters, and ask questions about the content, transforming passive listening into an interactive learning experience.

The AudioSeek Data Pipeline is an end-to-end, automated data processing solution designed to streamline the handling of audiobooks, transforming raw audio into a structured and queryable source of information. The pipeline is built on Apache Airflow, running in a Dockerized environment, for robust and scalable orchestration.

## Overview of Pipeline Architecture

This pipeline consists of a series of core steps, each defined as a task in a Directed Acyclic Graph (DAG) in Airflow. This allows for efficient scheduling, monitoring, and logging of the entire workflow. The DAGs are designed to automate the entire audiobook processing pipeline, from initial transcription to the final question-answering service.

**Project Flow Diagram:**

*(Placeholder for Project Flow.png)*

### Components

- **Scripts (`Data-Pipeline/scripts/`)**: A collection of Python scripts that perform the core data processing tasks, including transcription, chunking, embedding, and validation. Each script is designed to be run from the command line, making them easy to test and debug.
- **DAGs (`Data-Pipeline/dags/`)**: Airflow DAGs that define the workflow of the pipeline. The main DAG, `audio_processing_pipeline`, orchestrates the execution of the scripts in the correct order.
- **Services (`Data-Pipeline/services/`)**: Two FastAPI microservices that provide a real-time interface to the processed data. The `transcription` service provides an endpoint for transcribing audio files, while the `text_processing` service provides endpoints for chunking, embedding, and querying the data.
- **Data Storage**:
  - **SQLite:** A SQLite database (`audiobook_metadata.db`) is used to store metadata about the audiobooks, chapters, chunks, and entities.
  - **FAISS:** A FAISS index is used to store the vector embeddings of the text chunks, enabling efficient similarity search.
  - **File System:** The raw audio files, transcriptions, and other intermediate data are stored on the file system.

## DAG Overview and Status

Each stage of the pipeline is managed through a single DAG in Apache Airflow, enabling modularity and error isolation.

**Airflow UI Screenshot:**

*(Placeholder for a screenshot of the Airflow UI showing the `audio_processing_pipeline` DAG)*

## Detailed Workflow of Each Step

### 1. Model Validation

**Objective:** To ensure the transcription model is working correctly before processing the entire audiobook.

**Workflow:**
- **Input:** A ZIP file containing a sample audio file and a reference transcript.
- **Action:** The `validate_model.py` script is executed, which transcribes the sample audio file and compares the result to the reference transcript. The Word Error Rate (WER) and other metrics are calculated to measure the accuracy of the model.
- **Output:** A CSV file containing the validation summary.

**Script:** `scripts/validation/model_validation/validate_model.py`

### 2. Audio Transcription

**Objective:** To transcribe the audiobook files into text.

**Workflow:**
- **Input:** A ZIP file containing the audiobook chapters.
- **Action:** The `transcription.py` script is executed. This script uses the `faster-whisper` library to transcribe each audio file in the ZIP archive. The transcriptions are saved as individual text files, with timestamps for each segment.
- **Output:** A directory containing the transcribed text files and a summary CSV file.

**Script:** `scripts/transcription/transcription.py`

### 3. Cross-Model Validation

**Objective:** To compare the performance of different transcription models.

**Workflow:**
- **Input:** A small sample of the audiobook.
- **Action:** The sample is transcribed in parallel by `OpenAI Whisper` and `Wav2Vec2`. The results are then compared with the `faster-whisper` transcription. The `validate_transcription.py` script calculates the WER and ROUGE scores for each model.
- **Output:** A CSV file containing the cross-model validation summary.

**Scripts:**
- `scripts/validation/cross_model_evaluation/cross_model_sample_openaiwhisper.py`
- `scripts/validation/cross_model_evaluation/cross_model_sample_wav2vec.py`
- `scripts.validation/cross_model_evaluation/validate_transcription.py`

### 4. Text Chunking

**Objective:** To segment the transcribed text into smaller, manageable chunks.

**Workflow:**
- **Input:** The transcribed text files.
- **Action:** The `chunking_contentPOC.py` script parses the raw transcriptions into structured segments with timestamps. It then groups these segments into larger chunks of a specific token size, with an overlap to maintain context. The script also performs entity extraction using `spaCy`.
- **Output:** A JSON file containing the chunked text, along with metadata such as timestamps, token counts, and extracted entities.

**Script:** `scripts/chunking/chunking_contentPOC.py`

### 5. Embedding Generation

**Objective:** To create vector embeddings for the text chunks to enable semantic search.

**Workflow:**
- **Input:** The JSON file containing the chunked text.
- **Action:** The `embedding.py` script (currently a placeholder) would load the chunks and use a sentence-transformer model (`all-MiniLM-L6-v2`) to generate a vector embedding for each chunk. These embeddings are then stored in a FAISS index for efficient similarity search.
- **Output:** A FAISS index file.

**Script:** `scripts/embedding/embedding.py` (Placeholder)

## FastAPI Services

The project includes two FastAPI microservices for real-time interaction with the processed data.

### Transcription Service (`http://localhost:8000`)

- **`POST /transcribe`**: Upload an audio file to be transcribed.
  - **Request Body**: An audio file (`.mp3`, `.wav`, etc.).
  - **Response**: A JSON object containing the transcription with timestamps.

**Transcription Service Screenshot:**

*(Placeholder for a screenshot of the Transcription Service API documentation at http://localhost:8000/docs)*

### Text Processing Service (`http://localhost:8001`)

- **Chunking**
  - `POST /chunk`: Chunk a transcript file into segments with chapter detection and entity extraction.
- **Embedding**
  - `POST /embed`: Generate embeddings for text chunks.
- **Vector DB**
  - `POST /vector-db/add`: Add documents with embeddings to the vector database.
  - `POST /vector-db/search`: Search for similar vectors in the database using an embedding.
  - `POST /vector-db/query`: Query the vector database with text (embedding generated automatically).
  - `GET /vector-db/stats`: Get vector database statistics.
- **Pipeline**
  - `POST /process`: Run the combined chunking and embedding pipeline.
  - `POST /process-full`: Run the complete pipeline: chunk, embed, and add to vector DB.
- **Metadata DB**
  - `POST /metadata/audiobooks`: Create a new audiobook.
  - `POST /metadata/chapters`: Create a new chapter.
  - `POST /metadata/chunks`: Create a new chunk.
  - `POST /metadata/entities`: Create a new entity.
  - `POST /metadata/entity_mentions`: Create a new entity mention.
  - `GET /metadata/audiobooks/{audiobook_id}/chapters`: Get chapters for an audiobook.
  - `GET /metadata/chunks`: Get chunks for an audiobook, optionally filtered by chapter.
  - `GET /metadata/entities/{audiobook_id}`: Get entities for an audiobook.
- **QA**
  - `POST /qa/ask`: Answer questions about the audiobook.

**Text Processing Service Screenshot:**

*(Placeholder for a screenshot of the Text Processing Service API documentation at http://localhost:8001/docs)*

## Technology Stack and Dependencies

The AudioSeek pipeline relies on a variety of tools and libraries:

- **Orchestration:** Apache Airflow
- **Containerization:** Docker, Docker Compose
- **Python Libraries:**
  - `faster-whisper`: For audio transcription.
  - `sentence-transformers`: For generating text embeddings.
  - `faiss-cpu`: For the vector database.
  - `fastapi`: for creating the API services.
  - `uvicorn`: for running the FastAPI services.
  - `pandas`: For data manipulation.
  - `numpy`: For numerical operations.
  - `spacy`: For natural language processing, including entity extraction.
  - `psutil`: For monitoring system resource usage.
  - `SQLAlchemy`: For interacting with the SQLite database.

## Orchestration and Scheduling with Apache Airflow

Apache Airflow’s DAGs orchestrate each task, ensuring a well-coordinated and reliable data pipeline. The main DAG, `audio_processing_pipeline`, is manually triggered. The tasks in the DAG are connected in a specific order, with dependencies ensuring that a task only runs after the preceding tasks have completed successfully. The DAG also uses XComs to pass data between tasks.

**Log Screenshot:**

*(Placeholder for a screenshot of the Airflow logs for a DAG run)*

## Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- OpenAI API key
- SpaCy English model: `python -m spacy download en_core_web_sm`

### Running the Pipeline

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd audioSeek
    ```
2.  **Set up the environment:**
    - Create a `.env` file in the `Data-Pipeline` directory and add your OpenAI API key:
      ```
      API_KEY=your_llm_api_key
      ```
3.  **Run the services using Docker Compose:**
    ```bash
    cd Data-Pipeline
    docker-compose up -d
    ```
4.  **Access the services:**
    - Airflow UI: http://localhost:8080 (user: airflow, pass: airflow)
    - Transcription Service: http://localhost:8000
    - Text Processing Service: http://localhost:8001

## Conclusion

The AudioSeek Data Pipeline is a robust, modular, and scalable solution that automates the processing of audiobooks. The use of Airflow for orchestration and FastAPI for services provides a powerful platform for transforming audio content into a valuable and searchable knowledge base.