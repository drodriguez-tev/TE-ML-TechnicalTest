# Description

Here is my solution for the TE-ML-Technical Test, consisting of developing the following main components:

- Text from scanned document extractor.
- RAG for question answering based on PDF file.
- API to use these services.

# How to run with Docker

## Prerequisites

- Docker and docker-compose installed in your machine.

## 1. Create .env file

Create a new .env file based on the provided .env.example file.

Remember to add your Hugging Face API token so that all the required models are properly downloaded.

## 2. Run docker-compose up

Inside the root folder of the repository, run `docker-compose up --build` so that the required images are built and the created container is started.

The API should be listening in port 5000.

# Usage with Postman

## Prerequisites

- Postman.
- Docker container up and running.

## Submit a scanned document and extract the identified name!

1. Create a POST request to http://localhost:5000/upload.
2. In the body of the request, select `form-data` for the format.
3. Enter first_name, last_name, and file using the following key-value pairs format:
    
    
    | Key | Value |
    | --- | --- |
    | first_name | [ADD_HERE_THE_FIRST_NAME] |
    | last_name | [ADD_HERE_THE_SECOND_NAME] |
    | file | [APPEND YOUR SCANNED DOCUMENT FILE] |
    - NOTE: Valid formats for the file are .jpg, .jpeg, and .png.
4. Send the request. If everything went ok, you should be receiving a 200 response with the following content in JSON format:
    
    
    | first_name | Identified first name in the document |
    | --- | --- |
    | last_name | Identified last name in the document |
    | bounding_box_cords | List of lists with the 4 coordinates of each bounding point of the bounding box. |
    | similiarty_score | Similarity comparison score in percentage from 1 to 100 between the identified name and the name provided by the user.  |

## Ask a question about the provided contract!

1. Create a POST request to http://127.0.0.1:5000/ask.
2. In the body of the request, select `form-data` for the format.
3. Enter the question that you want the model to answer using the following key-value pairs format:
    
    
    | Key | Value |
    | --- | --- |
    | question | [THE QUESTION YOU WANT TO ASK] |
    - NOTE: The API only has access to the test contract provided via email. It is not designed to submit a different contract so that the user can ask questions about it.
4. Send the request. If everything went ok, you should be receiving a 200 response with the following content in JSON format:
    
    
    | question | The user’s submitted question. |
    | --- | --- |
    | response | The answer to the user’s question based on what was found in the document. |

# External libraries and frameworks

- Haystack: Opensource AI Framework for building multimodal AI.
- easyocr: Ready-to-use OCR library for simplifying OCR tasks.
- thefuzz: Library for fuzzy string matching.
- dotenv: Library for managing environment variables.
- spacy: NLP library used for NER tasks.
- pillow: Library used for image processing tasks.
- Flask: Framework for creating web apps and APIs in Python.