# KKBox Music Streaming Customer Churn Rate Prediction 
KKBOX Customer Churn Prediction using Machine Learning and Deep Learning Techniques, a comparison

Music streaming services are very popular modern-day, there are different types of music streaming services present today. There are market leaders such as Apple and Spotify and also there are many music streaming industries which market themselves to a niche market of customers. Due to the competitiveness in this industry, some of the medium and small-scale music streaming services find it difficult to maintain their customer base. Therefore these companies need to focus to retain their customer base. One such way is to use analytics to predict whether a customer would leave the music streaming service. If the customer would be predicted to leave, the marketing teams would improve their strategies to maintain their customer base.  Therefore one way to predict this would be to use a customer churn prediction model using the existing customer base. 
Customer churn prediction could be developed through several methods, one method would be to use a statistical model to predict if a customer would be churned. Another very popular method would be to use machine learning algorithms such as Random Forest and K- Nearest Neighbour to predict customer churn. Furthermore, customer churn prediction could be performed using Neural Networks and Deep Learning Models. This project focuses on using both Machine Learning and Neural Network models to predict the customer churn rate for music streaming services which focuses on catering to a niche market.

## Models Used

This project explores the use of Deep Learning models for churn prediction. The following models were implemented and evaluated:

### Deep Neural Network (DNN) Classifier

A DNN classifier was built using TensorFlow. The model architecture consists of two hidden layers with 30 and 10 units, respectively.

The performance of the DNN classifier on the test set was:
- Accuracy: 0.509

### Keras Sequential Model

A Keras sequential model was also implemented. The architecture includes:
- A Flatten layer
- Three Dense hidden layers with 512, 200, and 100 units respectively (using ReLU activation)
- A final Dense layer with 2 units and softmax activation for binary classification.

The performance of the Keras model during training was:
- Training Accuracy: 0.5001
- Validation Accuracy: 0.4994

## Model Comparison

A comparison of different machine learning and deep learning models was performed. The detailed results, including various performance metrics for each model, can be found in the `model comparison.xlsx` file.

The accuracy scores obtained from the notebook `DL-Experiments (1).ipynb` for the deep learning models are relatively low, suggesting further optimization or exploration of different architectures might be needed.


