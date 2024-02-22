# MOVIE-GENRE-CLASSIFICATION
**Discrete Info:**

Title: Predicting Movie Genres from Plot Summaries

Abstract:
In this project, we aim to develop a machine learning model capable of predicting the genre of a movie based on its plot summary. Leveraging text processing techniques such as TF-IDF (Term Frequency-Inverse Document Frequency) and various classification algorithms including Naive Bayes, Logistic Regression, and Support Vector Machines, we explore the effectiveness of different approaches in genre prediction. Our dataset consists of a collection of movie plot summaries and their corresponding genres, enabling us to train and evaluate the performance of the models. Through experimentation and evaluation, we seek to identify the most accurate and efficient model for genre classification, contributing to the field of natural language processing and movie genre prediction.

Introduction:
Movie genres play a significant role in the entertainment industry, helping viewers identify their preferences and aiding in recommendations. Predicting movie genres automatically from textual information such as plot summaries can enhance content recommendation systems and movie database organization. In this project, we delve into the domain of natural language processing and machine learning to develop a model that predicts movie genres based solely on plot summaries. By employing techniques like TF-IDF and classifiers such as Naive Bayes, Logistic Regression, and Support Vector Machines, we aim to create an accurate and efficient genre classification system.

Methodology:
1. Data Collection: We gather a diverse dataset comprising movie plot summaries and their corresponding genres from reputable sources or movie databases.
2. Preprocessing: The collected text data undergoes preprocessing steps including tokenization, stop-word removal, and stemming or lemmatization to enhance feature extraction.
3. Feature Extraction: We utilize TF-IDF (Term Frequency-Inverse Document Frequency) to convert the text data into numerical vectors, representing the importance of words in each document.
4. Model Selection and Training: We experiment with multiple classification algorithms including Naive Bayes, Logistic Regression, and Support Vector Machines to train genre prediction models.
5. Model Evaluation: The trained models are evaluated using metrics such as accuracy, precision, recall, and F1-score to assess their performance in predicting movie genres.
6. Hyperparameter Tuning: We perform hyperparameter tuning to optimize the performance of the selected models.
7. Comparison and Analysis: The performance of each model is compared and analyzed to identify the most effective approach for movie genre prediction.

Results and Discussion:
The experimental results demonstrate the effectiveness of the machine learning models in predicting movie genres from plot summaries. Among the evaluated classifiers, Support Vector Machines exhibit the highest accuracy and robustness in genre classification tasks. However, Logistic Regression also yields competitive results with faster training times. Naive Bayes, while simple and efficient, shows slightly lower accuracy compared to SVM and Logistic Regression. Additionally, we observe that the choice of text processing techniques and feature representations significantly influences the performance of the models. Overall, our findings highlight the potential of machine learning in automating movie genre prediction tasks and improving content recommendation systems.

Conclusion:
In this project, we developed a machine learning model capable of predicting movie genres from plot summaries. Through experimentation with various classifiers and text processing techniques, we identified Support Vector Machines as the most effective approach for genre classification. Our work contributes to the advancement of natural language processing and demonstrates the feasibility of automated movie genre prediction systems. Future research could explore advanced text embeddings and deep learning architectures for further improving genre prediction accuracy and scalability.
