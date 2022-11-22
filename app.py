from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
@app.route('/index.html')
def home():
    return render_template('index.html')

@app.route('/generic_data_analysis.html')
def generic():
    return render_template('generic_data_analysis.html')

@app.route('/playground.html')
def playground():
    return render_template('playground.html')

@app.route('/male-female-dist.html')
def male_female_age_dist():
    return render_template('male-female-dist.html')

@app.route('/countrywise_gender_responses.html')
def country_dist():
    return render_template('countrywise_gender_responses.html')

@app.route('/occupation_analysis.html')
def occupation_analysis():
    return render_template('occupation_analysis.html')

@app.route('/undergrad_major.html')
def undergrad_major():
    return render_template('undergrad_major.html')

@app.route('/bachelors_degree.html')
def bachelors_degree():
    return render_template('bachelors_degree.html')

@app.route('/closer_examination.html')
def closer_examination():
    return render_template('closer_examination.html')

if __name__ == '__main__':
   app.config["CACHE_TYPE"] = "null"
   app.run()