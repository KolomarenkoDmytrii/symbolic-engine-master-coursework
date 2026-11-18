"""Rule extraction and Prolog compilation module."""

from .models import Goal, Rule, RuleBatch, DocumentMetadata
from .extractor import build_chains, invoke_extractor
from .compiler import render_prolog, atomic_write_json, atomic_write_text
from .pipeline import process_file
