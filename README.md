# Sentiment Analysis on Movie Reviews

The sentiment analysis system utilizes an LSTM model to classify IMDb movie reviews as positive or negative. It preprocesses text, applies GloVe embeddings and trains the model achieving 85.39% accuracy.

## Execution Guide:

1. Run the following command line in the terminal:
     ```
     pip install pandas numpy matplotlib tensorflow scikit-learn
     ```

2. Copy the path of the `IMDb Dataset.csv` file and paste it in the code

3. Download the GloVe embeddings from the official website: **https://nlp.stanford.edu/projects/glove/** > Download the glove.6B.zip file > Extract the `glove.6B.100d.txt` file from the zip folder (or) Download the `glove.6B.100d.txt` from the repository

4. Copy and paste the directory of this `.txt` file into the code

5. After running all the cells of the code, in the last cell you can enter a review of your own choice and run the cell

6. The answer to the review will be provided by the cell with an accuracy of more than 80%

7. The output will look like:

   ![image](https://github.com/user-attachments/assets/bf57691c-c80c-4ba7-ba77-ba55f000715e)

## Overview:

This project performs sentiment analysis on IMDb movie reviews using an LSTM-based model. The model classifies reviews as either **positive** or **negative** based on the text content.

1. **Data Preprocessing**: The IMDb dataset is loaded, and text data is tokenized and converted into sequences of integers. These sequences are then padded to ensure uniform input length. The sentiment labels are mapped to binary values (1 for positive, 0 for negative).

2. **GloVe Embeddings**: Pre-trained GloVe word embeddings are used to convert words into dense vector representations. These embeddings are mapped to words in the dataset and stored in an embedding matrix.

3. **Model Architecture**: A Sequential LSTM model is created with an embedding layer (using the GloVe embeddings), an LSTM layer, and a Dense output layer with a sigmoid activation for binary classification. The model is compiled with Adam optimizer and binary cross-entropy loss.

4. **Model Training**: The model is trained for 5 epochs on the preprocessed training data, and training/validation accuracy and loss are visualized.

5. **Evaluation**: The model is evaluated on the test dataset, with metrics such as accuracy, precision, recall, and F1-score calculated. The model achieves an accuracy of 85.39%.

6. **Sentiment Prediction**: A function is implemented to predict the sentiment of new movie reviews. Three sample reviews are classified as either "positive" or "negative" based on the model's prediction.

This project demonstrates a robust approach to sentiment analysis using deep learning, specifically LSTM with word embeddings for better understanding of text sentiment.
