# copied from https://www.youtube.com/watch?v=Z1RJmh_OqeA

from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime


app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///test.db"
db = SQLAlchemy(app)

class Todo(db.Model):
    id_num = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String, nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return "<Task {}>".format(self.id_num)


@app.route("/", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        task_content = request.form["content_name"]
        new_task = Todo(content=task_content)

        try:
            db.session.add(new_task)
            db.session.commit()
            return redirect("/")
        except Exception as e:
            return "oops: {}".format(e)
    else:
        # return "HelloWorld"
        tasks = Todo.query.order_by(Todo.date_created).all()
        return render_template("index2.html", tasks=tasks)

@app.route("/delete/<int:id_num>")
def delete(id_num):
    task_to_delete = Todo.query.get_or_404(id_num)
    try:
        db.session.delete(task_to_delete)
        db.session.commit()
        return redirect("/")
    except Exception as e:
        return "oops: {}".format(e)

@app.route("/update/<int:id_num>", methods=["GET", "POST"])
def update(id_num):
    task = Todo.query.get_or_404(id_num)
    if request.method == "POST":
        new_task_content = request.form["content_name"]
        task.content = new_task_content

        try:
            # task object already referencing something in db, and it has been updated, but need to tell db about these changes
            db.session.commit()
            return redirect("/")
        except Exception as e:
            return "oops: {}".format(e)
    else:
        return render_template("update.html", task=task)


if __name__ == "__main__":
    app.run(debug=True)
