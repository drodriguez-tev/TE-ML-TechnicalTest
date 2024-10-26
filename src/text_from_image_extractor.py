from PIL import Image
import cv2
import numpy as np
import easyocr
import spacy
from dotenv import load_dotenv, dotenv_values, find_dotenv
from thefuzz import fuzz

class TextFromImageExtractor:
    """
    Class for representing the text from image extractor for extracting text from scans of ID documents.

    ...

    Attributes
    ----------
        raw_data_folder : str
            Path of folder where the images will be stored before processing.
        processed_data_folder : str
            Path of folder where the images will be stored after processing.

    Methods
    -------
        text_extraction_and_landmarking(image_path):
            Extracts the identified full name, first name, and last name from the image in 'image_path', along with its bounding box coordinates.
        get_similarity_score(extracted_result, real_result):
            Performs fuzzy matching and calculates a similarity score comparint 'extracted_result' and 'real_result'.
        extract_text(image_path):
            Identifies and extracts the text from the image in 'image_path' after preprocessing it.
        correct_skew(image_path):
            Corrects the skewness of the image in 'image_path' and stores the converted image in the processed data folder.
        get_angle(cv2_img):
            Gets the skewness angle from the cv2 image object.
        ner(text_and_boxes):
            Performs NER to extract the elements from 'text_and_boxes' keys that refer to persons. 
    """
    def __init__(self, raw_data_folder, processed_data_folder):
        """
        Constructs the attrbiutes for the TextFromImageExtractor class.

        Parameters
        ----------
            raw_data_folder : str
                Path of folder where the images will be stored before processing.
            processed_data_folder : str
                Path of folder where the images will be stored after processing.
        """
        self.raw_data_folder = raw_data_folder
        self.processed_data_folder = processed_data_folder
    
    def text_extraction_and_landmarking(self, image_path):
        """
        Extracts the identified full name, first name, and last name from the image in 'image_path', along with its bounding box coordinates.
        
        Parameters
        ----------
            image_path : str
                Path where the image from which to extract its text and bounding box coordinates is located.
        
        Returns
        -------
            result : dict
                Returns a dictionary with the identified full name, first name, last name, and bounding box coordinates.
        """

        # Get text and boxes
        text_and_boxes = self.extract_text(image_path)
        
        # Perform NER
        ner_results = self.ner(text_and_boxes)

        # Use first result as top result
        top_ner_result = ner_results[0]

        # Format results in dict format
        splitted_name = top_ner_result.split(" ")
        
        first_name = splitted_name[0]
        last_name = splitted_name[1]
        box = text_and_boxes[top_ner_result]
        result = {
            "full_name": top_ner_result,
            "first_name": first_name,
            "last_name": last_name,
            "box": box
        }

        return result

    def get_similarity_score(self, extracted_result, real_result):
        """
        Performs fuzzy matching and calculates a similarity score comparint 'extracted_result' and 'real_result'.

        Parameters
        ----------
            extracted_result : str
                The text that was extracted from calling the text_extraction_and_landmarking method.
            real_result : str
                The real text provided by the user
        
        Returns
        -------
            similarity_score : int
                Similarity percentage from comparing 'extraced_result' and 'real_result'
        
        """
        similarity_score = fuzz.ratio(extracted_result, real_result)
        return similarity_score

    def extract_text(self, image_path):
        """
        Identifies and extracts the text from the image in 'image_path' after preprocessing it.
        
        Parameters
        ----------
            image_path : str
                Path where the image from which to extract its text and bounding box coordinates is located. 

        Returns
        -------
            text_and_boxes : dict
                Dictionary with the identified text as keys and their respective bounding boxes as values.
        """
        # Correct any skewness in the image
        self.correct_skew(image_path)

        # Convert the image to grays
        image = cv2.imread(self.processed_data_folder + image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.bitwise_not(image)

        # Initialize OCR Reader
        reader = easyocr.Reader(['en'], gpu=False)

        # detect text on image
        text_ = reader.readtext(image)

        threshold = 0.25
        # draw bbox and text
        text_and_boxes = {}
        for t_, t in enumerate(text_):
            bbox, text, score = t
            for i in range(len(bbox)):
                for j in range(2):
                    bbox[i][j] = int(bbox[i][j])

            preprocessed_text = ''.join([i for i in text if (i.isalpha() or i==" ")]).title()

            if score > threshold:
                text_and_boxes[preprocessed_text] = bbox
                #cv2.rectangle(image, bbox[0], bbox[2], (0, 255, 0), 2)
                #text__.append(text)
        return text_and_boxes

    def correct_skew(self, image_path):
        """
        Corrects the skewness of the image in 'image_path' and stores the converted image in the processed data folder.

        Parameters
        ----------
            image_path : str
                Path where the image to correct its skewness is located.
        
        Returns
        -------
        NONE
        """

        img = cv2.imread(self.raw_data_folder + image_path)
        angle = self.get_angle(img)

        img = Image.open(self.raw_data_folder + image_path)
        if angle != 0:
            img = img.rotate(angle)
        process_path = self.processed_data_folder + image_path  
        img.save(process_path)

    def get_angle(self, cv2_img):
        """
        Gets the skewness angle from the cv2 image object.

        Parameters
        ----------
            cv2_img : Numpy Array
                Image in Numpy Array representation from using cv2.imread().
        
        Returns
        -------
            angle : float
                The detected skewness angle from the image.
        """
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)

        thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        return angle

    def ner(self, text_and_boxes):
        """
        Performs NER to extract the elements from 'text_and_boxes' keys that refer to persons. 

        Parameters
        ----------
            text_and_boxes : dict
                Dictionary containing the previously identified text and bounding box from the scanned document.

        Returns
        -------
            results : list
                List containing the identified text that refers to a person's name.
        
        """
        nlp = spacy.load("en_core_web_md")
        results = []
        for text in text_and_boxes.keys():
            document = nlp(text)
            for ent in document.ents:
                if ent.label_ == "PERSON" and len(ent.text.split(" ")) == 2:
                    results.append(ent.text)
        return results
