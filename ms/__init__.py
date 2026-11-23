"""MomentSearch, small enough to read: ask a question about your videos, get the moments that answer it."""
from .core import setup, VIDEOS
from .ingest import add_video, upload
from .show import ask, look_inside, library
from .evals import Q, report, which_branch
from .app import app
