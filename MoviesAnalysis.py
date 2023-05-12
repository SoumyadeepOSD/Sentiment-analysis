import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('moviesdata.csv')
df.head()
df["sentiment"].value_counts()
sentiment_label = df.sentiment.factorize()
tweet = df.review.values
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=5000)

tokenizer.fit_on_texts(tweet)
encoded_docs = tokenizer.texts_to_sequences(tweet)
from tensorflow.keras.preprocessing.sequence import pad_sequences

padded_sequence = pad_sequences(encoded_docs, maxlen=200)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding


embedding_vector_length = 32
model = Sequential()
model.add(Embedding(99273, embedding_vector_length, input_length=200))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
print(model.summary())
# history = model.fit(padded_sequence,sentiment_label[0],validation_split=0.2, epochs=5, batch_size=32)
import pickle
# with open('model_pickle_movie', 'wb') as f:
#     pickle.dump(model, f)
with open('model_pickle_movie', 'rb') as f:
    mp = pickle.load(f)

def predict_sentiment(text):
    tw = tokenizer.texts_to_sequences([text])
    tw = pad_sequences(tw,maxlen=200)
    prediction = int(mp.predict(tw).round().item())
    print("Predicted label: ", sentiment_label[1][prediction])
    st.write("Your text tone can be categorised as: ",sentiment_label[1][prediction])
    scores = sia.polarity_scores(text)
    a = scores["neg"]
    b = scores["pos"]
    c = scores["neu"]
    d = scores["compound"]
    st.write("Negativity is", a*100,"%")
    st.write("Positivity is", b*100,"%")
    st.write("Nutral is", c*100,"%")
    st.write("Compound is", d*100,"%")
    st.header(":orange[Graphical Representation]")
    labels = ['Negative', 'Positive', 'Neutral', 'compound']
    sizes = [a, b, c, abs(d)]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)


# test_sentence1 = "I liked the movie a lot...I want to watch more movies like that"
# predict_sentiment(test_sentence1)

# test_sentence2 = "This is the best chinese I ever had!"
# predict_sentiment(test_sentence2)

# test_sentence3 = "I'm not gonna look at you!"
# predict_sentiment(test_sentence3)

# test_sentence4 = "This is the best movie"
# predict_sentiment(test_sentence4)

# test_sentence5 = "This is an unexpected worst movie"
# predict_sentiment(test_sentence5)

import streamlit as st
st.title(":blue[Sentiment analysis app]")



test_sentence3 = st.text_input('Enter the text')
if st.button('Done'):
    predict_sentiment(test_sentence3)
