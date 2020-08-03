# FrontEnd


 server(multi_senti_fuc).py --> using Google Translate to convert any language into English & then find score for Text. Based on score it classified as Positive, Negative, Neutral.

	server.py --> In this, pickle file is used (pickle is generated in Training of Model --> final_model_multilangual.py) for predicting the Sentiment.

	Both server.py & server(multi_senti_fuc).py are Flask application.

 My Model is trained based on Hindi, English & Hinglish.

 In this both server.py & server(multi_senti_fuc).py has feature to take Input as csv & Predicting the sentiments then store the sentiments into csv file.
