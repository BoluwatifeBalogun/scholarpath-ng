"""ScholarPath NG application factory."""
import secrets
from datetime import timedelta
from functools import wraps

from flask import (Flask, abort, flash, g, redirect, render_template, request,
                   session, url_for)

from . import config, services
from .db import User, make_session
from .grading import GRADING, INSTITUTION_TYPES, TIER_MEANING, bands_table, label_for


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            flash("Sign in to continue.", "info")
            return redirect(url_for("auth.login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if g.user is None:
            flash("Sign in to continue.", "info")
            return redirect(url_for("auth.login", next=request.path))
        if not g.user.is_admin:
            abort(403)
        return view(*args, **kwargs)
    return wrapped


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=config.secret_key(),
        DATABASE_URL=config.database_url(),
        MAX_CONTENT_LENGTH=config.MAX_UPLOAD_MB * 1024 * 1024,
        PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SEED=True,
    )
    if test_config:
        app.config.update(test_config)

    Session, engine = make_session(app.config["DATABASE_URL"])
    app.db = Session
    app.engine = engine

    if app.config["SEED"]:
        services.seed(Session())
        Session.remove()

    @app.teardown_appcontext
    def close_session(_exc=None):
        Session.remove()

    @app.before_request
    def load_user_and_guard():
        g.db = Session()
        uid = session.get("uid")
        g.user = g.db.get(User, uid) if uid else None
        if "csrf" not in session:
            session["csrf"] = secrets.token_hex(16)
        if request.method == "POST":
            sent = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token")
            if not sent or not secrets.compare_digest(sent, session["csrf"]):
                abort(400, "Your session expired. Reload the page and try again.")

    @app.context_processor
    def inject():
        return {
            "app_name": config.APP_NAME, "app_tagline": config.APP_TAGLINE,
            "user": g.get("user"), "csrf_token": session.get("csrf", ""),
            "GRADING": GRADING, "INSTITUTION_TYPES": INSTITUTION_TYPES,
            "TIER_MEANING": TIER_MEANING, "bands_table": bands_table,
            "label_for": label_for,
        }

    @app.template_filter("pct")
    def pct(value, digits=1):
        return f"{float(value) * 100:.{digits}f}%"

    @app.template_filter("num")
    def num(value):
        return f"{int(value):,}"

    @app.template_filter("g")
    def gfmt(value):
        return f"{float(value):g}" if isinstance(value, (int, float)) else value

    @app.template_filter("when")
    def when(value):
        return value.strftime("%d %b %Y, %H:%M") if value else ""

    @app.errorhandler(400)
    def bad_request(e):
        return render_template("error.html", code=400, title="That request could not be read",
                               message=getattr(e, "description", "")), 400

    @app.errorhandler(403)
    def forbidden(_e):
        return render_template("error.html", code=403, title="Administrators only",
                               message="This page is for the system administrator. "
                                       "Sign in with an administrator account to open it."), 403

    @app.errorhandler(404)
    def not_found(_e):
        return render_template("error.html", code=404, title="Page not found",
                               message="The address may be mistyped, or the record was removed."), 404

    @app.errorhandler(413)
    def too_large(_e):
        return render_template("error.html", code=413, title="That file is too large",
                               message=f"Upload a CSV smaller than {config.MAX_UPLOAD_MB} MB."), 413

    from .routes import admin, auth, predict, public
    for bp in (public.bp, auth.bp, predict.bp, admin.bp):
        app.register_blueprint(bp)
    return app
