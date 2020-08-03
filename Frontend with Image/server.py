from flask import Flask, render_template, request,jsonify
app = Flask(__name__)
import pandas as pd

import pickle
with open('classifier_multi.pkl','rb') as f:
    classifier = pickle.load(f)
with open('CountVectorizer_multi.pkl','rb') as f:
    cv = pickle.load(f)


@app.route("/")
def view_template():
    return render_template("index.html")

@app.route("/data", methods=["GET","POST"])
def form_data():
    if request.method == "GET":
        return "<h1>Sorry, You mistaken somewhere</h1>"
    else:
        user_data = request.form   
        selected = user_data['selected']
        
        if int(selected)==1:
            text = user_data["text_area"]
            
            text = str(text).strip()
            print(text)
            input_data = [] 
            input_data.append(text)
            input_data = cv.transform(input_data).toarray()

            input_pred = classifier.predict(input_data)
            input_pred = input_pred.astype(int)
            
            if input_pred[0]==1:
                result = "Positive"
            elif input_pred[0]==0:
                result = "Neutral"
            else:
                result = "Negative"
            #print(result)            
            return jsonify(msg=str(result))
        
        if int(selected)==0:
            imge = user_data["img_name"]
            print(type(imge))
            #imge = "insert_file.csv"
            head=['text']
            dataset = pd.read_csv(imge,names=head,engine = "python")
            data = dataset['text'].tolist()
            score=[]
            for item in data:
                 text = str(item).strip()
                 #print(text)
                 input_data = [] 
                 input_data.append(text)
                 input_data = cv.transform(input_data).toarray()
                 input_pred = classifier.predict(input_data)
                 input_pred = input_pred.astype(int)
                 if input_pred[0]==1:
                     sc = "Positive"
                 elif input_pred[0]==0:
                     sc = "Neutral"
                 else:
                     sc = "Negative"
                 score.append(sc)
            result = ', '.join(score)
            print(score)
            return jsonify(msg=str(result))

        if int(selected)==2:
            image1 = user_data["img_name"]   
        
            import pytesseract
            
            pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
    
            from PIL import Image
           
            
            image = Image.open(image1)
            
            
            text = pytesseract.image_to_string(image)
            print(text)
            
            text = text.strip()
            trans_text = translation_to_eng(text)
            #text = "film is good lekin climax kharaab hai"
            #text = "film is good but climax kharaab hai"
            print(trans_text)
            trans_text=clean(trans_text) 
            #trans_text=clean_Lemi(text)
            
            text_token = tokenization(trans_text)
            text_pos = pos_mark(text_token)            
            wn_text_senti_score = senti_score(text_pos)            
            vander_text_senti_score = vader_senti_score(trans_text)
            adjusted_score1 = adjusted_score(wn_text_senti_score,vander_text_senti_score)
            print(adjusted_score1)  
            
            senti = Sentiment_value(adjusted_score1)
            result = senti
            print(result)
            
            return jsonify(msg=str(result))
            


if __name__ == "__main__":
  app.run(debug=True,use_reloader=False)