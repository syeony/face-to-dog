from flask import Flask, request, render_template
from utils import get_most_similar_dog

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        img = request.files['image']
        result = get_most_similar_dog(img)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
