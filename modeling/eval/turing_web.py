"""
Web interface for Human Turing Test evaluation.

Run with: python turing_web.py --input_files file1.jsonl file2.jsonl --num_examples 50 --port 5000
"""

import os
import json
import random
import csv
import argparse
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, session, jsonify

app = Flask(__name__, template_folder='templates')
app.secret_key = os.urandom(24)

# Global config
CONFIG = {
    'input_files': [],  # List of input file paths
    'output_dir': 'human_turing_results',
    'seed': 42,
    'base': False,
    'num_examples': None,  # Number of examples per file (None = all)
}

# Cache for loaded examples
EXAMPLES_CACHE = None


def extract_dialogue(text, ctx, source_name, base=False):
    """Extract dialogue from model output.
    
    Auto-detects SFT mode if '_sft_' is in the source filename.
    """
    # Auto-detect SFT mode based on filename
    is_sft = '_sft_' in source_name.lower()
    
    if is_sft:
        # Remove speaker labels (with and without trailing space)
        text = text.replace('Speaker 1: ', '').replace('Speaker 2: ', '')
        text = text.replace('Speaker 1:', '').replace('Speaker 2:', '').strip()
        return text
    
    last_utterance = ctx[-1].strip()
    if "<dialogue>" not in text:
        print(f"No dialogue tags found in text: {text}")
        return None
    
    if "</dialogue>" not in text:
        text = text.split("<dialogue>")[1].strip()
        text = text.replace('Speaker 1: ', '').replace('Speaker 2: ', '')
        text = text.replace('Speaker 1:', '').replace('Speaker 2:', '').strip()
        return text

    text = text.split("</dialogue>")[0].strip()
    if "<dialogue>" not in text:
        return None
    dialogue = text.split("<dialogue>")[1].strip()
    if "\n" in dialogue:
        dialogue = dialogue.split("\n")[0].strip()
    dialogue = dialogue.replace('Speaker 1: ', '').replace('Speaker 2: ', '')
    dialogue = dialogue.replace('Speaker 1:', '').replace('Speaker 2:', '').strip()
    
    if dialogue.lower() == "predicted next dialogue":
        return None
    
    if dialogue.lower() == last_utterance.lower():
        return None
    
    return dialogue


def load_examples():
    """Load and process examples from all JSONL files, interleave and shuffle between files."""
    global EXAMPLES_CACHE
    
    # Build cache key from all input files
    cache_key = f"{','.join(sorted(CONFIG['input_files']))}_{CONFIG['seed']}_{CONFIG['num_examples']}"
    
    if EXAMPLES_CACHE is not None and EXAMPLES_CACHE.get('_cache_key') == cache_key:
        return EXAMPLES_CACHE['examples']
    
    random.seed(CONFIG['seed'])
    
    # Store examples by file for interleaving
    examples_by_file = {}
    
    for input_file in CONFIG['input_files']:
        if not os.path.exists(input_file):
            print(f"Warning: Input file not found: {input_file}")
            continue
        
        source_name = Path(input_file).stem  # Use filename without extension as source identifier
        
        with open(input_file, 'r') as f:
            raw_examples = [json.loads(line) for line in f]
        
        file_examples = []
        
        for ex in raw_examples:
            model_dialogue = extract_dialogue(
                ex["completion"], 
                ex["context"], 
                source_name,
                CONFIG['base']
            )
            
            if model_dialogue is None:
                continue
            
            # Clean ground truth - remove speaker labels
            ground_truth = ex['true_response']
            ground_truth = ground_truth.replace('Speaker 1: ', '').replace('Speaker 2: ', '')
            ground_truth = ground_truth.replace('Speaker 1:', '').replace('Speaker 2:', '').strip()
            
            # Randomize positions
            gt_position = random.choice(['A', 'B'])
            model_position = 'B' if gt_position == 'A' else 'A'
            
            # Use raw text without speaker labels
            response_a = ground_truth if gt_position == 'A' else model_dialogue
            response_b = model_dialogue if model_position == 'B' else ground_truth
            
            file_examples.append({
                "sample_id": ex["sample_id"],
                "context": ex["context"],
                "ground_truth": ground_truth,
                "model_dialogue": model_dialogue,
                "gt_position": gt_position,
                "model_position": model_position,
                "response_a": response_a,
                "response_b": response_b,
                "source_file": source_name,  # Track which file this came from
            })
        
        examples_by_file[source_name] = file_examples
        print(f"Loaded {len(file_examples)} examples from {source_name}")
    
    # Combine all examples from all files
    all_examples = []
    for file_examples in examples_by_file.values():
        all_examples.extend(file_examples)
    
    # Randomly shuffle all examples together
    random.shuffle(all_examples)
    
    # Assign sequential indices after shuffling
    for i, ex in enumerate(all_examples):
        ex['index'] = i
    
    EXAMPLES_CACHE = {'examples': all_examples, '_cache_key': cache_key}
    
    # Print distribution verification
    file_counts = {}
    for ex in all_examples:
        src = ex['source_file']
        file_counts[src] = file_counts.get(src, 0) + 1
    print(f"Total examples after shuffling: {len(all_examples)}")
    print(f"Distribution by file: {file_counts}")
    
    return all_examples


def get_session_id():
    """Generate a unique session identifier based on input files."""
    names = sorted([Path(f).stem for f in CONFIG['input_files']])
    return "_".join(names[:3])  # Limit to avoid very long filenames


def get_csv_path(annotator_name):
    """Get the CSV file path for an annotator."""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in annotator_name)
    session_id = get_session_id()
    return os.path.join(CONFIG['output_dir'], f"{safe_name}_{session_id}.csv")


def get_completed_indices(annotator_name):
    """Get set of example indices already completed by annotator."""
    csv_path = get_csv_path(annotator_name)
    completed = set()
    
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(int(row['example_index']))
    
    return completed


def get_completed_counts_by_file(annotator_name):
    """Get count of completed annotations per source file."""
    csv_path = get_csv_path(annotator_name)
    counts = {}
    
    # Initialize all files with 0
    for input_file in CONFIG['input_files']:
        source_name = Path(input_file).stem
        counts[source_name] = 0
    
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                source = row.get('source_file', 'unknown')
                counts[source] = counts.get(source, 0) + 1
    
    return counts


def save_response(annotator_name, example_index, choice, example_data):
    """Save a single response to CSV, appending to file."""
    csv_path = get_csv_path(annotator_name)
    file_exists = os.path.exists(csv_path)
    
    # Determine winner based on choice
    if choice == 'Tie':
        winner = 'tie'
    elif choice == example_data['model_position']:
        winner = 'model'
    elif choice == example_data['gt_position']:
        winner = 'ground_truth'
    else:
        winner = 'invalid'
    
    row = {
        'example_index': example_index,
        'sample_id': example_data['sample_id'],
        'source_file': example_data['source_file'],  # Track source file
        'annotator': annotator_name,
        'choice': choice,
        'winner': winner,
        'gt_position': example_data['gt_position'],
        'model_position': example_data['model_position'],
        'timestamp': datetime.now().isoformat(),
        'context': json.dumps(example_data['context']),
        'response_a': example_data['response_a'],
        'response_b': example_data['response_b'],
        'ground_truth': example_data['ground_truth'],
        'model_dialogue': example_data['model_dialogue'],
    }
    
    fieldnames = list(row.keys())
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    return winner


def get_next_example_index(annotator_name, examples):
    """Get the next unanswered example index, respecting num_examples per file."""
    completed = get_completed_indices(annotator_name)
    completed_by_file = get_completed_counts_by_file(annotator_name)
    num_examples = CONFIG['num_examples']
    
    print(f"Completed: {len(completed)}")
    print(f"Completed by file: {completed_by_file}")
    print(f"Examples: {len(examples)}")
    
    for i in range(len(examples)):
        if i in completed:
            continue
        
        # Check if this file has reached its quota
        source_file = examples[i]['source_file']
        if num_examples is not None and completed_by_file.get(source_file, 0) >= num_examples:
            # This file has enough annotations, skip
            continue
        
        return i
    
    return None  # All done


def get_progress(annotator_name, examples):
    """Get progress stats for annotator, respecting num_examples per file."""
    completed_by_file = get_completed_counts_by_file(annotator_name)
    num_examples = CONFIG['num_examples']
    
    # Count available examples per file
    available_by_file = {}
    for ex in examples:
        source = ex['source_file']
        available_by_file[source] = available_by_file.get(source, 0) + 1
    
    # Calculate total target and completed
    total_target = 0
    total_completed = 0
    
    for source in available_by_file:
        available = available_by_file[source]
        completed = completed_by_file.get(source, 0)
        
        if num_examples is not None:
            # Target is min of num_examples and available
            target = min(num_examples, available)
            # Completed is min of completed and target (cap at target)
            completed = min(completed, target)
        else:
            target = available
        
        total_target += target
        total_completed += completed
    
    return {
        'completed': total_completed,
        'total': total_target,
        'percentage': round(total_completed / total_target * 100, 1) if total_target > 0 else 0
    }


def get_stats_by_file(annotator_name, examples):
    """Get stats broken down by source file."""
    csv_path = get_csv_path(annotator_name)
    num_examples = CONFIG['num_examples']
    
    # Count available examples per file
    available_by_file = {}
    for ex in examples:
        source = ex['source_file']
        available_by_file[source] = available_by_file.get(source, 0) + 1
    
    stats_by_file = {}
    
    # Initialize stats for all expected files
    for input_file in CONFIG['input_files']:
        source_name = Path(input_file).stem
        available = available_by_file.get(source_name, 0)
        target = min(num_examples, available) if num_examples is not None else available
        
        stats_by_file[source_name] = {
            'model_wins': 0.0,
            'gt_wins': 0.0,
            'ties': 0,
            'total': 0,  # completed count
            'target': target,  # target number of annotations
            'available': available,  # total available examples
        }
    
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                source = row.get('source_file', 'unknown')
                if source not in stats_by_file:
                    stats_by_file[source] = {
                        'model_wins': 0.0,
                        'gt_wins': 0.0,
                        'ties': 0,
                        'total': 0,
                        'target': 0,
                        'available': 0,
                    }
                
                stats_by_file[source]['total'] += 1
                
                if row['winner'] == 'model':
                    stats_by_file[source]['model_wins'] += 1
                elif row['winner'] == 'ground_truth':
                    stats_by_file[source]['gt_wins'] += 1
                elif row['winner'] == 'tie':
                    stats_by_file[source]['ties'] += 1
                    stats_by_file[source]['model_wins'] += 0.5
                    stats_by_file[source]['gt_wins'] += 0.5
    
    # Calculate percentages and progress
    for source, stats in stats_by_file.items():
        total_score = stats['model_wins'] + stats['gt_wins']
        stats['model_pct'] = round(stats['model_wins'] / total_score * 100, 1) if total_score > 0 else 0
        stats['gt_pct'] = round(stats['gt_wins'] / total_score * 100, 1) if total_score > 0 else 0
        stats['progress_pct'] = round(stats['total'] / stats['target'] * 100, 1) if stats['target'] > 0 else 0
    
    return stats_by_file


@app.route('/')
def index():
    """Landing page - ask for annotator name."""
    num_files = len(CONFIG['input_files'])
    file_names = [Path(f).stem for f in CONFIG['input_files']]
    num_examples = CONFIG['num_examples']
    return render_template('login.html', num_files=num_files, file_names=file_names, num_examples=num_examples)


@app.route('/start', methods=['POST'])
def start():
    """Start the annotation session."""
    annotator_name = request.form.get('annotator_name', '').strip()
    if not annotator_name:
        return redirect(url_for('index'))
    
    session['annotator'] = annotator_name
    return redirect(url_for('annotate'))


@app.route('/annotate')
def annotate():
    """Main annotation page."""
    annotator = session.get('annotator')
    if not annotator:
        return redirect(url_for('index'))
    
    try:
        examples = load_examples()
    except FileNotFoundError as e:
        return render_template('error.html', message=str(e))
    
    if not examples:
        return render_template('error.html', message="No valid examples found in input files")
    
    progress = get_progress(annotator, examples)
    next_idx = get_next_example_index(annotator, examples)
    
    if next_idx is None:
        return render_template('complete.html', 
                             annotator=annotator, 
                             progress=progress,
                             csv_path=get_csv_path(annotator))
    
    example = examples[next_idx]
    
    # Format context for display
    formatted_context = []
    for i, utt in enumerate(example['context']):
        speaker = f"Speaker {(i % 2) + 1}"
        formatted_context.append({'speaker': speaker, 'text': utt})
    
    return render_template('annotate.html',
                         annotator=annotator,
                         example=example,
                         example_index=next_idx,
                         formatted_context=formatted_context,
                         progress=progress)


@app.route('/submit', methods=['POST'])
def submit():
    """Submit an annotation."""
    annotator = session.get('annotator')
    if not annotator:
        return jsonify({'error': 'Not logged in'}), 401
    
    data = request.json
    example_index = data.get('example_index')
    choice = data.get('choice')
    
    if example_index is None or choice not in ['A', 'B', 'Tie']:
        return jsonify({'error': 'Invalid submission'}), 400
    
    examples = load_examples()
    if example_index < 0 or example_index >= len(examples):
        return jsonify({'error': 'Invalid example index'}), 400
    
    example = examples[example_index]
    winner = save_response(annotator, example_index, choice, example)
    
    progress = get_progress(annotator, examples)
    next_idx = get_next_example_index(annotator, examples)
    
    return jsonify({
        'success': True,
        'winner': winner,
        'progress': progress,
        'next_index': next_idx,
        'is_complete': next_idx is None
    })


@app.route('/stats')
def stats():
    """Show stats for current annotator, broken down by source file."""
    annotator = session.get('annotator')
    if not annotator:
        return redirect(url_for('index'))
    
    examples = load_examples()
    progress = get_progress(annotator, examples)
    
    # Get stats by file
    stats_by_file = get_stats_by_file(annotator, examples)
    
    # Calculate overall stats
    total_model_wins = sum(s['model_wins'] for s in stats_by_file.values())
    total_gt_wins = sum(s['gt_wins'] for s in stats_by_file.values())
    total_ties = sum(s['ties'] for s in stats_by_file.values())
    total_answered = sum(s['total'] for s in stats_by_file.values())
    
    total_score = total_model_wins + total_gt_wins
    overall_model_pct = round(total_model_wins / total_score * 100, 1) if total_score > 0 else 0
    overall_gt_pct = round(total_gt_wins / total_score * 100, 1) if total_score > 0 else 0
    
    return render_template('stats.html',
                         annotator=annotator,
                         progress=progress,
                         stats_by_file=stats_by_file,
                         total_model_wins=total_model_wins,
                         total_gt_wins=total_gt_wins,
                         total_ties=total_ties,
                         total_answered=total_answered,
                         overall_model_pct=overall_model_pct,
                         overall_gt_pct=overall_gt_pct)


@app.route('/logout')
def logout():
    """Logout and return to login page."""
    session.pop('annotator', None)
    global EXAMPLES_CACHE
    EXAMPLES_CACHE = None  # Clear cache
    return redirect(url_for('index'))


def create_templates():
    """Create HTML templates directory and files."""
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(template_dir, exist_ok=True)
    
    # Base template
    base_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Turing Test{% endblock %}</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e4e4e7;
            line-height: 1.6;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .header .subtitle {
            color: #a1a1aa;
            margin-top: 10px;
        }
        
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }
        
        .progress-bar {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            height: 12px;
            overflow: hidden;
            margin: 15px 0;
        }
        
        .progress-fill {
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
        }
        
        .progress-text {
            text-align: center;
            color: #a1a1aa;
            font-size: 0.9rem;
        }
        
        .btn {
            display: inline-block;
            padding: 14px 32px;
            font-size: 1rem;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.2s ease;
            text-decoration: none;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        }
        
        .btn-secondary {
            background: rgba(255,255,255,0.1);
            color: #e4e4e7;
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .btn-secondary:hover {
            background: rgba(255,255,255,0.15);
        }
        
        .btn-choice {
            width: 100%;
            padding: 20px;
            margin: 10px 0;
            text-align: left;
            background: rgba(255,255,255,0.03);
            border: 2px solid rgba(255,255,255,0.1);
        }
        
        .btn-choice:hover {
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.1);
        }
        
        .btn-choice.selected {
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.2);
        }
        
        input[type="text"] {
            width: 100%;
            padding: 14px 20px;
            font-size: 1rem;
            border: 2px solid rgba(255,255,255,0.1);
            border-radius: 10px;
            background: rgba(255,255,255,0.05);
            color: #e4e4e7;
            margin-bottom: 20px;
            transition: border-color 0.2s;
        }
        
        input[type="text"]:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .context-box {
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .utterance {
            padding: 12px 16px;
            margin: 8px 0;
            border-radius: 12px;
            max-width: 85%;
        }
        
        .speaker-1 {
            background: rgba(102, 126, 234, 0.2);
            border-left: 3px solid #667eea;
        }
        
        .speaker-2 {
            background: rgba(118, 75, 162, 0.2);
            border-left: 3px solid #764ba2;
            margin-left: auto;
        }
        
        .speaker-label {
            font-size: 0.75rem;
            font-weight: 600;
            color: #a1a1aa;
            margin-bottom: 4px;
        }
        
        .response-box {
            padding: 20px;
            border-radius: 12px;
            margin: 15px 0;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .response-a {
            background: rgba(59, 130, 246, 0.1);
            border: 2px solid rgba(59, 130, 246, 0.3);
        }
        
        .response-b {
            background: rgba(236, 72, 153, 0.1);
            border: 2px solid rgba(236, 72, 153, 0.3);
        }
        
        .response-box:hover {
            transform: translateY(-2px);
        }
        
        .response-a:hover, .response-a.selected {
            border-color: #3b82f6;
            box-shadow: 0 5px 20px rgba(59, 130, 246, 0.2);
        }
        
        .response-b:hover, .response-b.selected {
            border-color: #ec4899;
            box-shadow: 0 5px 20px rgba(236, 72, 153, 0.2);
        }

        .response-tie {
            background: rgba(168, 85, 247, 0.1);
            border: 2px solid rgba(168, 85, 247, 0.3);
        }

        .response-tie:hover, .response-tie.selected {
            border-color: #a855f7;
            box-shadow: 0 5px 20px rgba(168, 85, 247, 0.2);
        }

        .response-tie .response-label { color: #a855f7; }

        .response-label {
            font-weight: 700;
            font-size: 1.1rem;
            margin-bottom: 8px;
        }
        
        .response-a .response-label { color: #3b82f6; }
        .response-b .response-label { color: #ec4899; }
        
        .question {
            text-align: center;
            font-size: 1.2rem;
            font-weight: 600;
            margin: 30px 0 20px;
            color: #f4f4f5;
        }
        
        .actions {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 30px;
        }
        
        .nav-links {
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-top: 20px;
        }
        
        .nav-links a {
            color: #a1a1aa;
            text-decoration: none;
            font-size: 0.9rem;
        }
        
        .nav-links a:hover {
            color: #667eea;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 20px 0;
        }
        
        .stat-card {
            background: rgba(255,255,255,0.03);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .stat-label {
            color: #a1a1aa;
            font-size: 0.9rem;
            margin-top: 5px;
        }
        
        .feedback {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 25px;
            border-radius: 10px;
            font-weight: 600;
            animation: slideIn 0.3s ease;
            z-index: 1000;
        }
        
        .feedback.success {
            background: rgba(34, 197, 94, 0.9);
            color: white;
        }
        
        @keyframes slideIn {
            from { transform: translateX(100px); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        .loading {
            opacity: 0.6;
            pointer-events: none;
        }
        
        .example-counter {
            text-align: center;
            color: #a1a1aa;
            margin-bottom: 20px;
            font-size: 0.9rem;
        }
        
        .file-stats-card {
            background: rgba(255,255,255,0.03);
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
        }
        
        .file-stats-card h4 {
            color: #f4f4f5;
            margin-bottom: 15px;
            font-size: 1.1rem;
        }
        
        .file-stats-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-size: 0.9rem;
        }
        
        .file-stats-label {
            color: #a1a1aa;
        }
        
        .file-stats-value {
            color: #e4e4e7;
            font-weight: 600;
        }
        
        .file-list {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        
        .file-tag {
            background: rgba(102, 126, 234, 0.2);
            color: #667eea;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
        }
    </style>
</head>
<body>
    {% block content %}{% endblock %}
    {% block scripts %}{% endblock %}
</body>
</html>'''
    
    # Login template
    login_html = '''{% extends "base.html" %}
{% block title %}Login - Turing Test{% endblock %}
{% block content %}
<div class="container">
    <div class="header">
        <h1>🤖 Human Turing Test</h1>
        <p class="subtitle">Help us evaluate AI-generated dialogue</p>
    </div>
    
    <div class="card" style="max-width: 500px; margin: 0 auto;">
        <h2 style="margin-bottom: 20px; text-align: center;">Welcome, Annotator!</h2>
        <p style="color: #a1a1aa; margin-bottom: 25px; text-align: center;">
            Please enter your name to begin or continue your annotation session.
        </p>
        
        <form action="{{ url_for('start') }}" method="POST">
            <input type="text" name="annotator_name" placeholder="Enter your name" required autofocus>
            
            <div style="margin-bottom: 20px; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 10px;">
                <p style="color: #a1a1aa; font-size: 0.9rem; margin-bottom: 10px;">
                    Evaluating {{ num_files }} model(s){% if num_examples %} ({{ num_examples }} examples each){% endif %}:
                </p>
                <div class="file-list">
                    {% for name in file_names %}
                    <span class="file-tag">{{ name }}</span>
                    {% endfor %}
                </div>
            </div>
            
            <button type="submit" class="btn btn-primary" style="width: 100%;">
                Start Annotating
            </button>
        </form>
    </div>
</div>
{% endblock %}'''
    
    # Annotate template
    annotate_html = '''{% extends "base.html" %}
{% block title %}Annotate - Turing Test{% endblock %}
{% block content %}
<div class="container">
    <div class="header">
        <h1>🤖 Human Turing Test</h1>
        <p class="subtitle">Welcome, {{ annotator }}</p>
    </div>
    
    <div class="card">
        <div class="progress-bar">
            <div class="progress-fill" style="width: {{ progress.percentage }}%"></div>
        </div>
        <p class="progress-text">{{ progress.completed }} / {{ progress.total }} completed ({{ progress.percentage }}%)</p>
    </div>
    
    <div class="card" id="annotation-card">
        <p class="example-counter">Example #{{ example_index + 1 }}</p>
        
        <h3 style="margin-bottom: 15px; color: #f4f4f5;">Conversation Context:</h3>
        <div class="context-box">
            {% for item in formatted_context %}
            <div class="utterance speaker-{{ (loop.index0 % 2) + 1 }}">
                <div class="speaker-label">{{ item.speaker }}</div>
                {{ item.text }}
            </div>
            {% endfor %}
        </div>
        
        <p class="question">Which response is more natural and appropriate?</p>
        
        <div class="response-box response-a" onclick="selectResponse('A')" id="response-a">
            <div class="response-label">Response A</div>
            <p>{{ example.response_a }}</p>
        </div>
        
        <div class="response-box response-b" onclick="selectResponse('B')" id="response-b">
            <div class="response-label">Response B</div>
            <p>{{ example.response_b }}</p>
        </div>

        <div class="response-box response-tie" onclick="selectResponse('Tie')" id="response-tie">
            <div class="response-label">Tie</div>
            <p>Both responses are equally good</p>
        </div>

        <div class="actions">
            <button class="btn btn-primary" id="submit-btn" onclick="submitChoice()" disabled>
                Submit Choice
            </button>
        </div>
    </div>
    
    <div class="nav-links">
        <a href="{{ url_for('stats') }}">View Statistics</a>
        <a href="{{ url_for('logout') }}">Logout</a>
    </div>
</div>

<div class="feedback success" id="feedback" style="display: none;">
    Saved! Loading next...
</div>
{% endblock %}

{% block scripts %}
<script>
    let selectedChoice = null;
    const exampleIndex = {{ example_index }};
    
    function selectResponse(choice) {
        selectedChoice = choice;

        document.getElementById('response-a').classList.remove('selected');
        document.getElementById('response-b').classList.remove('selected');
        document.getElementById('response-tie').classList.remove('selected');
        document.getElementById('response-' + choice.toLowerCase()).classList.add('selected');
        document.getElementById('submit-btn').disabled = false;
    }
    
    async function submitChoice() {
        if (!selectedChoice) return;
        
        const card = document.getElementById('annotation-card');
        const btn = document.getElementById('submit-btn');
        
        card.classList.add('loading');
        btn.disabled = true;
        btn.textContent = 'Saving...';
        
        try {
            const response = await fetch('{{ url_for("submit") }}', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    example_index: exampleIndex,
                    choice: selectedChoice
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                const feedback = document.getElementById('feedback');
                feedback.style.display = 'block';
                
                setTimeout(() => {
                    if (data.is_complete) {
                        window.location.href = '{{ url_for("stats") }}';
                    } else {
                        window.location.reload();
                    }
                }, 500);
            } else {
                alert('Error: ' + data.error);
                card.classList.remove('loading');
                btn.disabled = false;
                btn.textContent = 'Submit Choice';
            }
        } catch (error) {
            alert('Network error. Please try again.');
            card.classList.remove('loading');
            btn.disabled = false;
            btn.textContent = 'Submit Choice';
        }
    }
    
    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        if (e.key === 'a' || e.key === 'A' || e.key === '1') {
            selectResponse('A');
        } else if (e.key === 'b' || e.key === 'B' || e.key === '2') {
            selectResponse('B');
        } else if (e.key === 't' || e.key === 'T' || e.key === '3') {
            selectResponse('Tie');
        } else if (e.key === 'Enter' && selectedChoice) {
            submitChoice();
        }
    });
</script>
{% endblock %}'''
    
    # Complete template
    complete_html = '''{% extends "base.html" %}
{% block title %}Complete - Turing Test{% endblock %}
{% block content %}
<div class="container">
    <div class="header">
        <h1>🎉 All Done!</h1>
        <p class="subtitle">Thank you, {{ annotator }}!</p>
    </div>
    
    <div class="card" style="text-align: center;">
        <h2 style="margin-bottom: 20px;">You've completed all examples!</h2>
        <p style="color: #a1a1aa; margin-bottom: 20px;">
            Your annotations have been saved to:<br>
            <code style="background: rgba(0,0,0,0.3); padding: 5px 10px; border-radius: 5px;">{{ csv_path }}</code>
        </p>
        
        <div class="actions">
            <a href="{{ url_for('stats') }}" class="btn btn-primary">View Statistics</a>
            <a href="{{ url_for('logout') }}" class="btn btn-secondary">Logout</a>
        </div>
    </div>
</div>
{% endblock %}'''
    
    # Stats template - updated to show per-file stats
    stats_html = '''{% extends "base.html" %}
{% block title %}Statistics - Turing Test{% endblock %}
{% block content %}
<div class="container">
    <div class="header">
        <h1>📊 Your Statistics</h1>
        <p class="subtitle">{{ annotator }}'s annotation progress</p>
    </div>
    
    <div class="card">
        <h3 style="margin-bottom: 15px;">Overall Progress</h3>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {{ progress.percentage }}%"></div>
        </div>
        <p class="progress-text">{{ progress.completed }} / {{ progress.total }} completed ({{ progress.percentage }}%)</p>
    </div>
    
    <div class="card">
        <h3 style="margin-bottom: 20px;">Overall Results (ties count as 0.5 each)</h3>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{{ total_model_wins }}</div>
                <div class="stat-label">Model Score ({{ overall_model_pct }}%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ total_ties }}</div>
                <div class="stat-label">Ties</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{{ total_gt_wins }}</div>
                <div class="stat-label">Human Score ({{ overall_gt_pct }}%)</div>
            </div>
        </div>

        {% if total_answered > 0 %}
        <div style="margin-top: 20px;">
            <div style="display: flex; height: 30px; border-radius: 8px; overflow: hidden;">
                <div style="width: {{ overall_model_pct }}%; background: linear-gradient(90deg, #667eea, #764ba2);"></div>
                <div style="width: {{ overall_gt_pct }}%; background: linear-gradient(90deg, #3b82f6, #06b6d4);"></div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 10px; font-size: 0.85rem; color: #a1a1aa;">
                <span>Model: {{ overall_model_pct }}%</span>
                <span>Human: {{ overall_gt_pct }}%</span>
            </div>
        </div>
        {% endif %}
    </div>
    
    <div class="card">
        <h3 style="margin-bottom: 20px;">Results by Model/File</h3>
        {% for file_name, file_stats in stats_by_file.items() %}
        <div class="file-stats-card">
            <h4>{{ file_name }}</h4>
            <div class="file-stats-row">
                <span class="file-stats-label">Progress</span>
                <span class="file-stats-value">{{ file_stats.total }} / {{ file_stats.target }} ({{ file_stats.progress_pct }}%)</span>
            </div>
            <div style="margin-bottom: 12px;">
                <div class="progress-bar" style="height: 8px; margin: 5px 0;">
                    <div class="progress-fill" style="width: {{ file_stats.progress_pct }}%"></div>
                </div>
            </div>
            <div class="file-stats-row">
                <span class="file-stats-label">Model Score</span>
                <span class="file-stats-value">{{ file_stats.model_wins }} ({{ file_stats.model_pct }}%)</span>
            </div>
            <div class="file-stats-row">
                <span class="file-stats-label">Human Score</span>
                <span class="file-stats-value">{{ file_stats.gt_wins }} ({{ file_stats.gt_pct }}%)</span>
            </div>
            <div class="file-stats-row">
                <span class="file-stats-label">Ties</span>
                <span class="file-stats-value">{{ file_stats.ties }}</span>
            </div>
            {% if file_stats.total > 0 %}
            <div style="margin-top: 10px;">
                <div style="display: flex; height: 16px; border-radius: 4px; overflow: hidden;">
                    <div style="width: {{ file_stats.model_pct }}%; background: linear-gradient(90deg, #667eea, #764ba2);"></div>
                    <div style="width: {{ file_stats.gt_pct }}%; background: linear-gradient(90deg, #3b82f6, #06b6d4);"></div>
                </div>
            </div>
            {% endif %}
        </div>
        {% endfor %}
    </div>
    
    <div class="actions">
        {% if progress.completed < progress.total %}
        <a href="{{ url_for('annotate') }}" class="btn btn-primary">Continue Annotating</a>
        {% endif %}
        <a href="{{ url_for('logout') }}" class="btn btn-secondary">Logout</a>
    </div>
</div>
{% endblock %}'''
    
    # Error template
    error_html = '''{% extends "base.html" %}
{% block title %}Error - Turing Test{% endblock %}
{% block content %}
<div class="container">
    <div class="header">
        <h1>⚠️ Error</h1>
    </div>
    
    <div class="card" style="text-align: center;">
        <p style="color: #ef4444; margin-bottom: 20px;">{{ message }}</p>
        <a href="{{ url_for('index') }}" class="btn btn-primary">Go Back</a>
    </div>
</div>
{% endblock %}'''
    
    templates = {
        'base.html': base_html,
        'login.html': login_html,
        'annotate.html': annotate_html,
        'complete.html': complete_html,
        'stats.html': stats_html,
        'error.html': error_html,
    }
    
    for name, content in templates.items():
        path = os.path.join(template_dir, name)
        with open(path, 'w') as f:
            f.write(content)
    
    print(f"Created templates in {template_dir}")


def main():
    parser = argparse.ArgumentParser(description='Web interface for Human Turing Test')
    parser.add_argument('--input_files', type=str, nargs='+', required=True,
                        help='Paths to input JSONL files with generations (one or more)')
    parser.add_argument('--num_examples', type=int, default=None,
                        help='Target number of annotations per file; already annotated examples are skipped (default: all)')
    parser.add_argument('--output_dir', type=str, default='human_turing_results',
                        help='Directory to save annotation CSVs')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to run the web server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind the server')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible position randomization')
    parser.add_argument('--base', action='store_true',
                        help='Use base model extraction mode')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Update config
    CONFIG['input_files'] = args.input_files
    CONFIG['num_examples'] = args.num_examples
    CONFIG['output_dir'] = args.output_dir
    CONFIG['seed'] = args.seed
    CONFIG['base'] = args.base
    
    # Create templates
    create_templates()
    
    # Validate input files
    valid_files = []
    for input_file in args.input_files:
        if not os.path.exists(input_file):
            print(f"Warning: Input file not found: {input_file}")
        else:
            valid_files.append(input_file)
            print(f"Found: {input_file}")
    
    if not valid_files:
        print("Error: No valid input files found!")
        return
    
    CONFIG['input_files'] = valid_files
    
    print(f"\n{'='*50}")
    print(f"Starting Human Turing Test Web Interface")
    print(f"{'='*50}")
    print(f"Input files: {len(valid_files)}")
    for f in valid_files:
        print(f"  - {Path(f).name}")
    print(f"Examples per file: {args.num_examples or 'all'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"{'='*50}\n")
    
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
