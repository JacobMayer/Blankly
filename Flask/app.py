from flask import Flask, redirect, url_for, render_template, request, send_from_directory
import json

app = Flask(__name__,  static_url_path='')

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/assets/<path:path>')
def asset(path):
    print(path)
    return send_from_directory('assets', path)

@app.route("/login", methods=['GET', 'POST'])
def login():
        if request.method == 'POST':
            username = request.form.get('username') 
            password = request.form.get('password')

            data = {}
            with open("users.json", "r") as read_file:
                data = json.load(read_file)
        
            if data['username'] == username and data['password'] == password:
                return render_template("index.html")

        return render_template("pages-login.html")

@app.route("/register",  methods=['GET', 'POST'])
def register():
    message = ''
    
    if request.method == 'POST':
        user = {
            "username" : request.form.get('username'), 
            "email" : request.form.get('email'),
            "password" : request.form.get('password')
        }
        
        with open("users.json", "w") as outfile:
            json.dump(user, outfile)
  

    return render_template('pages-register.html', message=message)

if __name__ == "__main__":
    app.run(debug=True)
