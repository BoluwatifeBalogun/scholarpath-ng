import re

from flask import (Blueprint, flash, g, redirect, render_template, request,
                   session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

from ..db import User

bp = Blueprint("auth", __name__)
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
OCCUPATIONS = ["Student", "Academic officer", "Lecturer"]


def _safe_next(target):
    return target if target and target.startswith("/") and not target.startswith("//") else None


@bp.route("/register", methods=["GET", "POST"])
def register():
    if g.user:
        return redirect(url_for("predict.new"))
    errors, form = {}, {}
    if request.method == "POST":
        form = {k: request.form.get(k, "").strip() for k in ("name", "email", "occupation")}
        password = request.form.get("password", "")
        if len(form["name"]) < 2:
            errors["name"] = "Enter your full name."
        if not EMAIL.match(form["email"]):
            errors["email"] = "Enter an email address like name@school.edu.ng."
        elif g.db.query(User).filter_by(email=form["email"].lower()).first():
            errors["email"] = "An account with this email already exists. Sign in instead."
        if form["occupation"] not in OCCUPATIONS:
            errors["occupation"] = "Choose what describes you."
        if len(password) < 8:
            errors["password"] = "Use at least 8 characters."
        if not errors:
            user = User(name=form["name"], email=form["email"].lower(),
                        password_hash=generate_password_hash(password),
                        occupation=form["occupation"])
            g.db.add(user)
            g.db.commit()
            session.clear()
            session["uid"] = user.id
            session.permanent = True
            flash("Account created.", "success")
            return redirect(url_for("predict.new"))
    return render_template("register.html", errors=errors, form=form,
                           occupations=OCCUPATIONS), (422 if errors else 200)


@bp.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(url_for("predict.new"))
    error, email = None, ""
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user = g.db.query(User).filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, request.form.get("password", "")):
            nxt = _safe_next(request.args.get("next"))
            session.clear()
            session["uid"] = user.id
            session.permanent = True
            return redirect(nxt or url_for("admin.overview" if user.is_admin else "predict.new"))
        error = "That email and password do not match an account. Check both and try again."
    return render_template("login.html", error=error, email=email), (401 if error else 200)


@bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("Signed out.", "info")
    return redirect(url_for("public.home"))
