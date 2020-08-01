from flask import Flask, render_template, request,jsonify
app = Flask(__name__)

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
            


if __name__ == "__main__":
  app.run(debug=True,use_reloader=False)