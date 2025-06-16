from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import joblib
import random
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for session

# Load data and model
users = pd.read_csv("users.csv")
marks = pd.read_csv("student_marks.csv")
model = joblib.load("student_model.pkl")

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        name = request.form["name"]
        rollno = request.form["rollno"]
        match = users[(users["Name"] == name) & (users["RollNo"].astype(str) == rollno)]
        if not match.empty:
            session["name"] = name
            session["rollno"] = rollno
            return redirect(url_for("dashboard"))
        else:
            return "<h3>❌ Invalid login. <a href='/'>Try again</a></h3>"
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "name" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", name=session["name"], rollno=session["rollno"])

@app.route("/cgpa")
def cgpa():
    if "name" not in session:
        return redirect(url_for("login"))
    rollno = session["rollno"]
    name = session["name"]
    student = marks[marks["RollNo"].astype(str) == rollno]
    if student.empty:
        return "<h3>No data found for this student. <a href='/dashboard'>Back</a></h3>"
    X = student[["Math1", "Physics1", "Chem1", "CGPA1", "Math2", "Physics2", "Chem2", "CGPA2"]]
    pred = model.predict(X)[0]
    try:
        prediction = round(float(pred), 2)
    except (ValueError, TypeError):
        prediction = str(pred)
    student_data = student.iloc[0].to_dict()
    return render_template("cgpa.html", name=name, rollno=rollno, student=student_data, prediction=prediction)

@app.route("/graph")
def graph():
    if "name" not in session:
        return redirect(url_for("login"))
    rollno = session["rollno"]
    student = marks[marks["RollNo"].astype(str) == rollno]
    if student.empty:
        return "<h3>No data found for this student. <a href='/dashboard'>Back</a></h3>"
    student_data = student.iloc[0].to_dict()
    return render_template("graph.html", student=student_data)

@app.route("/tree")
def tree():
    if "name" not in session:
        return redirect(url_for("login"))
    
    fig = plt.figure(figsize=(10, 6))
    plot_tree(model, filled=True, feature_names=["Math1", "Physics1", "Chem1", "CGPA1", "Math2", "Physics2", "Chem2", "CGPA2"])
    plt.tight_layout()
    tree_path = "static/tree_plot.png"
    plt.savefig(tree_path)
    plt.close(fig)
    
    cache_buster = random.randint(1, 100000)
    return render_template("tree.html", tree_path=tree_path, cache_buster=cache_buster)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
