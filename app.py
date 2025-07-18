from flask import Flask, request, render_template
from utils import get_most_similar_dog
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        img = request.files['image']
        result = get_most_similar_dog(img)
    return render_template('index.html', result=result)

# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)