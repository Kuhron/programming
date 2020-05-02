from flask import Flask, request, render_template, Response, redirect, url_for
import requests
app = Flask(__name__)


@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)


@app.route("/", methods=['GET', 'POST'])
def index():
    # https://stackoverflow.com/questions/19794695/flask-python-buttons
    print("request.form = {}".format(request.form.to_dict()))
    if True:#form.validate_on_submit():
        if 'download' in request.form:
            print("got download")
        elif 'watch' in request.form:
            print("got watch")

    # https://stackoverflow.com/questions/37890972/flask-user-input-and-buttons-with-back-end-actions-returning-text-to-user
    if request.method == "GET":
        return render_template("index.html")

    if "button_a" in request.form:
        text_1 = request.form["text_1"]
        text_2 = request.form["text_2"]
        success = len(text_1) % 2 == 0 and len(text_2) % 3 == 1

        return render_template("index.html", responseA="Successful" if success else "Failed")

    elif "button_b1" in request.form:
        success = True
        return render_template("index.html", responseB="Successful" if success else "Failed")

    elif "button_b2" in request.form:
        success = False
        return render_template("index.html", responseB="Successful" if success else "Failed", responseA="you messed this part up too")

    return render_template("index.html")

# @app.route("/<name>")
# def hello_name(name):
#     return "Hello {}".format(name)

@app.route('/cool_form', methods=['GET', 'POST'])
def cool_form():
    # https://stackoverflow.com/questions/27539309/how-do-i-create-a-link-to-another-html-page/27540234
    if request.method == 'POST':
        # do stuff when the form is submitted

        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('index'))

    # show the form, it wasn't submitted
    return render_template('cool_form.html')




if __name__ == "__main__":
    app.run(debug=True)
