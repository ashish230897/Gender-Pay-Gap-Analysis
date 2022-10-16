from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
@app.route('/index.html')
def home():
    return render_template('index.html')

@app.route('/male-female-dist.html')
def male_female_dist():
    return render_template('male-female-dist.html')

if __name__ == '__main__':
   app.run()