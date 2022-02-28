from flask import Flask, redirect, url_for, render_template, request, send_from_directory, make_response
import json
import os
from uuid import uuid4

app = Flask(__name__,  static_url_path='')

def getUserFromToken(token):
    if (os.path.isfile('users.json')):
        with open("users.json", "r") as read_file:
            data = json.load(read_file)
            for user in data['accounts']:
             if ('token' in user.keys()):
              if token == user['token'] and token != "":
                  return user['username']
    else:
        return "Sign in"

@app.route("/")
def home():
    token = request.cookies.get('token')
    if (os.path.isfile('users.json')):
        user = getUserFromToken(token)
        if user:
            return render_template("index.html", user=getUserFromToken(token))
        


    return redirect("/login")

@app.route('/signout')
def logout():
    token = request.cookies.get('token')
    with open("users.json", "r") as read_file:
        data = json.load(read_file)
        for user in data['accounts']:
         if ('token' in user.keys()):
          if token == user['token']:
            resp = make_response(redirect('/login'))
            resp.set_cookie('token', "")
            return resp


@app.route('/assets/<path:path>')
def asset(path):
    print(path)
    return send_from_directory('assets', path)

@app.route("/login", methods=['GET', 'POST'])
def login():
        if request.method == 'POST':

            if not (os.path.isfile('users.json')):
                return redirect("/register")

            username = request.form.get('username') 
            password = request.form.get('password')

            data = {}
            with open("users.json", "r") as read_file:
                data = json.load(read_file)
        

            for user in data['accounts']:
                if user['username'] == username and user['password'] == password:
                       resp = make_response(redirect('/'))
                       rand_token = str(uuid4())
                       resp.set_cookie('token', rand_token)
                       user['token'] = rand_token
                       with open("users.json", "w") as out_file:
                            json.dump(data, out_file)
                       return resp

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
        
        if (os.path.isfile('users.json')):
         database = json.load(open("users.json", "r"))            
         with open("users.json", "r") as outfile:
            for users in database['accounts']:
                if (users['username'] == user['username']):
                    return
         os.remove("users.json")
         database['accounts'].append(user)
         with open("users.json", "w") as outfile:
            json.dump(database, outfile)
        
            return redirect("/login")
            
        else: 
            with open("users.json", "w") as outfile:
                database = {"accounts" : [user]}
                json.dump(database, outfile)
            return redirect("/login")

    return render_template('pages-register.html', message=message)

@app.route("/faq")
def faq():
    token = request.cookies.get('token')
    return render_template("pages-faq.html", user=getUserFromToken(token))

if __name__ == "__main__":
    app.run(debug=True)
