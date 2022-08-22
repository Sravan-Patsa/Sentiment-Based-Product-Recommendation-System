from flask import Flask, render_template, request
import model

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommendations", methods = ['POST'])
def recommendations():
    username = str(request.form.get('username'))
    
    print('username ', username)
    if not username:
        return render_template("index.html")

    products, rating = model.getRecommendations(username)

    if rating == None:
        return render_template('index.html', error = products)

    product_rating = zip(products, rating)

    return render_template('recommendations.html', username = username, products = product_rating)

if __name__ == "__main__":
    app.run(debug = True)

