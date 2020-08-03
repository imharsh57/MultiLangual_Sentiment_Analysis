from flask import Flask, render_template, request,jsonify
app = Flask(__name__)

from defination import *
import pandas as pd

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

        
        if int(selected)==0:
            imge = user_data["img_name"]
            print(type(imge))
            #imge = "insert_file.csv"
            head=['text']
            dataset = pd.read_csv(imge,names=head,engine = "python")
           # print(dataset)
            data = dataset['text'].tolist()
            
            score=[]
            for item in data:
                 text = str(item).strip()
                 trans_text = translation_to_eng(text)
                 #print(trans_text)
                 trans_text=clean(trans_text)
                 text_token = tokenization(trans_text)
                 text_pos = pos_mark(text_token)            
                 wn_text_senti_score = senti_score(text_pos)            
                 vander_text_senti_score = vader_senti_score(trans_text)
                 adjusted_score1 = adjusted_score(wn_text_senti_score,vander_text_senti_score)
                 senti = Sentiment_value(adjusted_score1)
                 print(senti) 
                 score.append(senti)
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
  app.run(debug=True,use_reloader=False,port=8000)