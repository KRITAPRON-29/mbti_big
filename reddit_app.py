"""
Reddit Community MBTI Analyzer
Scrapes Reddit communities via .json endpoint (no API key needed),
predicts MBTI types using parallel NLP processing,
and streams results in real-time via SSE.
"""

import re
import uuid
import json
import time
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests as http_requests
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, stream_with_context

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Import type profiles for text predictor mode
from mbti_types import TYPE_PROFILES

# ========== Load Pre-trained Models ==========
print("Loading MBTI models...")
models = joblib.load("mbti_dimension_models.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
print("Models loaded successfully!")

# ========== In-memory Job Storage ==========
jobs = {}
jobs_lock = threading.Lock()

# ========== MBTI Nicknames ==========
TYPE_NICKNAMES = {
    "INTJ": "The Architect", "INTP": "The Thinker",
    "ENTJ": "The Commander", "ENTP": "The Debater",
    "INFJ": "The Advocate", "INFP": "The Mediator",
    "ENFJ": "The Protagonist", "ENFP": "The Campaigner",
    "ISTJ": "The Logistician", "ISFJ": "The Defender",
    "ESTJ": "The Executive", "ESFJ": "The Consul",
    "ISTP": "The Virtuoso", "ISFP": "The Adventurer",
    "ESTP": "The Entrepreneur", "ESFP": "The Entertainer",
}

# ========== Bias Correction Weights ==========
# Calibrated from actual per-class model accuracy (avg 66%):
#   EI: I=68% E=60% acc=65%  |  NS: N=75% S=73% acc=74%
#   TF: F=63% T=63% acc=63%  |  JP: P=64% J=58% acc=62%
# Weight = opposing_accuracy / this_accuracy (moderate — avoid all 4 dims pushing same direction)
BIAS_CORRECTION = {
    "EI": {"E": 1.13, "I": 0.87},   # กลาง ๆ: ลด I bias พอประมาณ
    "NS": {"N": 0.95, "S": 1.05},   # ลด N bias เล็กน้อย (75 vs 73)
    "TF": {"T": 1.00, "F": 1.00},   # Balanced เพราะ acc เท่ากันที่ 63%
    "JP": {"J": 1.10, "P": 0.90},   # กลาง ๆ: ลด P bias พอประมาณ
}

# ========== MBTI Dimension Info (for Text Predictor) ==========
DIMENSION_INFO = {
    "EI": {
        "E": {"name": "Extraversion", "emoji": "🗣️", "desc": "คุณได้พลังจากการมีปฏิสัมพันธ์กับผู้อื่น ชอบสังคม กระตือรือร้น และชอบแสดงออก"},
        "I": {"name": "Introversion", "emoji": "📚", "desc": "คุณได้พลังจากการอยู่คนเดียว ชอบคิดลึกซึ้ง สงบ และมีสมาธิสูง"},
    },
    "NS": {
        "N": {"name": "Intuition", "emoji": "🔮", "desc": "คุณมองภาพรวมและความเป็นไปได้ ชอบจินตนาการ สร้างสรรค์ และมองไปข้างหน้า"},
        "S": {"name": "Sensing", "emoji": "🔍", "desc": "คุณให้ความสำคัญกับข้อเท็จจริงและรายละเอียด ใส่ใจปัจจุบัน เป็นคนจริงจัง"},
    },
    "TF": {
        "T": {"name": "Thinking", "emoji": "🧠", "desc": "คุณตัดสินใจด้วยตรรกะและเหตุผล วิเคราะห์อย่างเป็นระบบ ยุติธรรม"},
        "F": {"name": "Feeling", "emoji": "💖", "desc": "คุณตัดสินใจด้วยความรู้สึกและคุณค่า เห็นอกเห็นใจ ใส่ใจผู้อื่น"},
    },
    "JP": {
        "J": {"name": "Judging", "emoji": "📋", "desc": "คุณชอบวางแผนและจัดระเบียบ มีเป้าหมายชัดเจน ตัดสินใจเด็ดขาด"},
        "P": {"name": "Perceiving", "emoji": "🎨", "desc": "คุณยืดหยุ่นและเปิดรับสิ่งใหม่ ปรับตัวเก่ง ชอบความเป็นธรรมชาติ"},
    },
}

# ========== Stop Words (English, built-in) ==========
STOP_WORDS = set("""
a about above after again against all am an and any are aren't as at be because been before being
below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down
during each few for from further get got had hadn't has hasn't have haven't having he he'd he'll
he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is
isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or
other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't
so some such than that that's the their theirs them themselves then there there's these they
they'd they'll they're they've this those through to too under until up upon very was wasn't we
we'd we'll we're we've were weren't what what's when when's where where's which while who who's
whom why why's will with won't would wouldn't you you'd you'll you're you've your yours yourself
yourselves just like really also even still think know get going go got one way people think
things thing make want much something well good new would could also like really think know
right going go got one way people think things thing make want much something well good new
see say said says time first two also back now even give way most find here much many still
every day made get use may take come could people part long great little make just know
think time back way also even give find made get use take come could part long great little
deleted removed edit amp https http www com reddit post thread subreddit lol lmao
""".split())

# ========== Text Cleaning Pipeline ==========
def clean_text(text):
    """Remove URLs, special chars, lowercase, strip extra whitespace."""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)  # Remove markdown links
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def remove_stop_words(text):
    """Remove stop words and return cleaned text + word list."""
    words = text.split()
    filtered = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return " ".join(filtered), filtered


def extract_subreddit_name(url_or_name):
    """Extract subreddit name from URL or raw name."""
    url_or_name = url_or_name.strip().rstrip("/")
    # Match patterns like /r/subredditname or r/subredditname
    match = re.search(r"(?:reddit\.com)?/?r/([a-zA-Z0-9_]+)", url_or_name)
    if match:
        return match.group(1)
    # If it's just a name
    if re.match(r"^[a-zA-Z0-9_]+$", url_or_name):
        return url_or_name
    return None


# ========== Reddit Scraper ==========
SCRAPE_HEADERS = {
    "User-Agent": "python:mbti.analyzer:v2.0 (educational project)",
}

REDDIT_BASE = "https://old.reddit.com"

SORT_MODES = ["hot", "new", "top", "rising"]
TOP_TIME_FILTERS = ["day", "week", "month", "year", "all"]


def _safe_reddit_get(url, params, max_retries=3):
    """Make a GET request to Reddit with retry logic and rate-limit handling."""
    for attempt in range(max_retries):
        try:
            resp = http_requests.get(url, headers=SCRAPE_HEADERS, params=params, timeout=20)
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"Rate limited, waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            if resp.status_code == 200:
                return resp
            # Handle redirect (302/301) — Reddit sometimes redirects
            if resp.status_code in (301, 302):
                print(f"Redirect {resp.status_code} for {url}")
                return None
            print(f"HTTP {resp.status_code} for {url}")
            return None
        except http_requests.exceptions.Timeout:
            print(f"Timeout (attempt {attempt+1}) for {url}")
            time.sleep(5)
        except http_requests.exceptions.ConnectionError:
            print(f"Connection error (attempt {attempt+1}) for {url}")
            time.sleep(8)
        except Exception as e:
            print(f"Request error: {e}")
            return None
    return None


def scrape_reddit_page(subreddit, sort="hot", after=None, limit=25, time_filter=None):
    """
    Scrape a single page of posts from a subreddit.
    Returns (posts_list, next_after_token).
    """
    url = f"{REDDIT_BASE}/r/{subreddit}/{sort}.json"
    params = {"limit": limit, "raw_json": 1}
    if after:
        params["after"] = after
    if sort == "top" and time_filter:
        params["t"] = time_filter
    
    resp = _safe_reddit_get(url, params)
    if not resp:
        return [], None
    
    try:
        data = resp.json()
    except Exception:
        return [], None
    
    children = data.get("data", {}).get("children", [])
    next_after = data.get("data", {}).get("after")
    
    posts = []
    for child in children:
        post_data = child.get("data", {})
        
        if post_data.get("stickied", False):
            continue
        
        title = post_data.get("title", "")
        selftext = post_data.get("selftext", "")
        combined = f"{title} {selftext}".strip()
        
        posts.append({
            "id": post_data.get("id", ""),
            "author": post_data.get("author", "[deleted]"),
            "title": title,
            "selftext": selftext,
            "text": combined,
            "score": post_data.get("score", 0),
            "num_comments": post_data.get("num_comments", 0),
            "permalink": post_data.get("permalink", ""),
            "created_utc": post_data.get("created_utc", 0),
        })
    
    return posts, next_after


def _extract_all_comments(node):
    """Recursively extract ALL comment texts from a Reddit comment tree."""
    comments = []
    if isinstance(node, dict):
        kind = node.get("kind")
        data = node.get("data", {})
        if kind == "t1":  # Comment
            body = data.get("body", "")
            author = data.get("author", "[deleted]")
            if body and body not in ("[deleted]", "[removed]") and author != "AutoModerator":
                comments.append({
                    "author": author,
                    "body": body,
                    "score": data.get("score", 0),
                })
            # Recurse into replies
            replies = data.get("replies")
            if isinstance(replies, dict):
                comments.extend(_extract_all_comments(replies))
        elif kind == "Listing":
            for child in data.get("children", []):
                comments.extend(_extract_all_comments(child))
        elif kind == "more":
            pass  # Skip "load more" stubs
    elif isinstance(node, list):
        for item in node:
            comments.extend(_extract_all_comments(item))
    return comments


def scrape_post_comments(permalink):
    """
    Scrape ALL comments from a single post.
    Returns list of {author, body, score}.
    """
    if not permalink:
        return []
    url = f"{REDDIT_BASE}{permalink}.json"
    params = {"raw_json": 1, "limit": 500, "depth": 10, "sort": "new"}
    
    resp = _safe_reddit_get(url, params)
    if not resp:
        return []
    
    try:
        data = resp.json()
    except Exception:
        return []
    
    if not isinstance(data, list) or len(data) < 2:
        return []
    
    # data[1] contains the comment tree
    return _extract_all_comments(data[1])


# ========== Parallel MBTI Prediction ==========
def apply_bias_correction(dim_key, classes, probas):
    """Apply bias correction weights to raw probabilities and re-normalize."""
    weights = BIAS_CORRECTION.get(dim_key, {})
    corrected = []
    for i, cls in enumerate(classes):
        w = weights.get(cls, 1.0)
        corrected.append(float(probas[i]) * w)
    # Re-normalize so they sum to 1.0
    total = sum(corrected)
    if total > 0:
        corrected = [p / total for p in corrected]
    return corrected


def predict_mbti_single(text_cleaned, vectorizer_ref, models_ref):
    """Predict MBTI for a single text. Thread-safe since models are read-only."""
    text_vec = vectorizer_ref.transform([text_cleaned])
    
    mbti_type = ""
    dim_details = {}
    
    for dim_key in ["EI", "NS", "TF", "JP"]:
        model = models_ref[dim_key]
        raw_probas = model.predict_proba(text_vec)[0]
        classes = model.classes_
        
        # Apply bias correction
        corrected_probas = apply_bias_correction(dim_key, classes, raw_probas)
        
        # Pick prediction from corrected probabilities
        best_idx = corrected_probas.index(max(corrected_probas))
        prediction = classes[best_idx]
        
        dim_probs = {cls: corrected_probas[i] for i, cls in enumerate(classes)}
        dim_details[dim_key] = {
            "prediction": prediction,
            "probabilities": dim_probs
        }
        mbti_type += prediction
    
    return mbti_type, dim_details


def predict_posts_parallel(posts, word_counter, max_workers=4):
    """
    Predict MBTI for a list of posts using parallel processing.
    Returns (results_list, updated_word_counter).
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_post = {}
        
        for post in posts:
            cleaned = clean_text(post["text"])
            cleaned_no_stop, words = remove_stop_words(cleaned)
            word_counter.update(words)
            
            if len(cleaned_no_stop.split()) >= 3:
                future = executor.submit(
                    predict_mbti_single, cleaned, vectorizer, models
                )
                future_to_post[future] = post
        
        for future in as_completed(future_to_post):
            post = future_to_post[future]
            try:
                mbti_type, dim_details = future.result(timeout=30)
                results.append({
                    "author": post["author"],
                    "title": post.get("title", ""),
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "permalink": post.get("permalink", ""),
                    "mbti_type": mbti_type,
                    "nickname": TYPE_NICKNAMES.get(mbti_type, "Unknown"),
                    "dimensions": dim_details,
                    "is_comment": post.get("is_comment", False),
                })
            except Exception as e:
                print(f"Prediction error for '{post.get('author','?')}': {e}")
    
    return results, word_counter


# ========== Background Analysis Job (Continuous) ==========
def compute_aggregates(all_results, word_counter):
    """Compute aggregate stats from all results."""
    type_counter = Counter(r["mbti_type"] for r in all_results)
    
    dim_counts = {"E": 0, "I": 0, "N": 0, "S": 0, "T": 0, "F": 0, "J": 0, "P": 0}
    for r in all_results:
        for letter in r["mbti_type"]:
            dim_counts[letter] = dim_counts.get(letter, 0) + 1
    
    total = len(all_results) or 1
    dim_ratios = {
        "EI": {"E": round(dim_counts["E"] / total * 100, 1), "I": round(dim_counts["I"] / total * 100, 1)},
        "NS": {"N": round(dim_counts["N"] / total * 100, 1), "S": round(dim_counts["S"] / total * 100, 1)},
        "TF": {"T": round(dim_counts["T"] / total * 100, 1), "F": round(dim_counts["F"] / total * 100, 1)},
        "JP": {"J": round(dim_counts["J"] / total * 100, 1), "P": round(dim_counts["P"] / total * 100, 1)},
    }
    
    top_type = type_counter.most_common(1)[0] if type_counter else ("N/A", 0)
    top_words = word_counter.most_common(30)
    
    # Get full profiles for top 3 MBTI types
    top3 = type_counter.most_common(3)
    top3_profiles = []
    for t_code, t_count in top3:
        prof = TYPE_PROFILES.get(t_code, {})
        mkt = prof.get("marketing", {})
        top3_profiles.append({
            "type": t_code,
            "count": t_count,
            "pct": round(t_count / len(all_results) * 100, 1) if all_results else 0,
            "nickname": TYPE_NICKNAMES.get(t_code, ""),
            "emoji": prof.get("emoji", "🧠"),
            "traits": prof.get("traits", ""),
            "strengths": prof.get("strengths", []),
            "weaknesses": prof.get("weaknesses", []),
            "communication": prof.get("communication", ""),
            "work_style": prof.get("work_style", ""),
            "careers": prof.get("careers", []),
            "marketing": {
                "approach": mkt.get("approach", ""),
                "do": mkt.get("do", []),
                "dont": mkt.get("dont", []),
                "content_style": mkt.get("content_style", ""),
                "buying_trigger": mkt.get("buying_trigger", ""),
            } if mkt else None,
        })
    
    top_profile = TYPE_PROFILES.get(top_type[0], {})
    marketing_data = top_profile.get("marketing", {})
    
    return {
        "type_distribution": dict(type_counter),
        "dim_ratios": dim_ratios,
        "top_type": {
            "type": top_type[0],
            "count": top_type[1],
            "nickname": TYPE_NICKNAMES.get(top_type[0], ""),
            "emoji": top_profile.get("emoji", "🧠"),
            "traits": top_profile.get("traits", ""),
        },
        "top_word": top_words[0] if top_words else ["N/A", 0],
        "word_frequencies": top_words,
        "top3_profiles": top3_profiles,
        "marketing": {
            "approach": marketing_data.get("approach", ""),
            "do": marketing_data.get("do", []),
            "dont": marketing_data.get("dont", []),
            "content_style": marketing_data.get("content_style", ""),
            "buying_trigger": marketing_data.get("buying_trigger", ""),
        } if marketing_data else None,
    }


def run_analysis_job(job_id, subreddit):
    """Run continuous analysis pipeline — loops through sort modes, scrapes posts + ALL comments."""
    with jobs_lock:
        jobs[job_id] = {
            "status": "scraping",
            "subreddit": subreddit,
            "events": [],
            "completed": False,
            "stop": False,
        }
    
    def add_event(event_type, data):
        with jobs_lock:
            jobs[job_id]["events"].append({
                "type": event_type,
                "data": data,
                "timestamp": time.time()
            })
    
    def should_stop():
        with jobs_lock:
            return jobs[job_id].get("stop", False)
    
    # Persistent state across rounds
    seen_ids = set()
    seen_comment_ids = set()
    all_results = []
    comment_count = 0
    word_counter = Counter()
    round_num = 0
    consecutive_empty = 0
    
    add_event("status", {"message": f"กำลังเริ่มดึงข้อมูลจาก r/{subreddit} (โพสต์ + คอมเมนต์)...", "phase": "scraping"})
    
    # Build list of (sort, time_filter) combos to cycle through
    sort_queue = []
    sort_queue.append(("hot", None))
    sort_queue.append(("new", None))
    sort_queue.append(("rising", None))
    for tf in TOP_TIME_FILTERS:
        sort_queue.append(("top", tf))
    
    sort_idx = 0
    after_tokens = {}  # (sort, time_filter) -> after token for pagination
    
    while not should_stop():
        sort_mode, time_filter = sort_queue[sort_idx % len(sort_queue)]
        queue_key = f"{sort_mode}_{time_filter or 'none'}"
        after = after_tokens.get(queue_key)
        
        round_num += 1
        sort_label = sort_mode
        if time_filter:
            sort_label = f"top/{time_filter}"
        
        add_event("status", {
            "message": f"🔄 Round {round_num} — ดึงโพสต์จาก r/{subreddit} ({sort_label})...",
            "phase": "scraping",
            "sort_mode": sort_label,
            "round": round_num,
        })
        
        # Scrape one page of posts
        posts, next_after = scrape_reddit_page(
            subreddit, sort=sort_mode, after=after, limit=25, time_filter=time_filter
        )
        
        if should_stop():
            break
        
        # Save pagination token
        if next_after:
            after_tokens[queue_key] = next_after
        else:
            after_tokens[queue_key] = None
        
        # Deduplicate posts
        new_posts = [p for p in posts if p["id"] and p["id"] not in seen_ids]
        for p in new_posts:
            seen_ids.add(p["id"])
        
        if not new_posts:
            consecutive_empty += 1
            sort_idx += 1
            if round_num == 1 and consecutive_empty >= 1:
                add_event("error", {
                    "message": f"ไม่สามารถดึงข้อมูลจาก r/{subreddit} ได้ — อาจไม่มี subreddit นี้ หรือถูก rate limit"
                })
                with jobs_lock:
                    jobs[job_id]["completed"] = True
                return
            
            add_event("status", {
                "message": f"⏭️ ไม่มีโพสต์ใหม่จาก {sort_label} — สลับโหมด...",
                "phase": "switching",
            })
            time.sleep(3)
            continue
        
        consecutive_empty = 0
        
        # ---- Scrape ALL comments for each new post ----
        all_comment_items = []
        for pidx, post in enumerate(new_posts):
            if should_stop():
                break
            
            add_event("status", {
                "message": f"💬 ดึงคอมเมนต์จากโพสต์ {pidx+1}/{len(new_posts)}: {post['title'][:50]}...",
                "phase": "scraping_comments",
            })
            
            raw_comments = scrape_post_comments(post["permalink"])
            
            for c in raw_comments:
                # Create a unique ID for each comment to deduplicate
                c_id = f"{post['id']}_{c['author']}_{hash(c['body'][:100])}"
                if c_id in seen_comment_ids:
                    continue
                seen_comment_ids.add(c_id)
                
                all_comment_items.append({
                    "id": c_id,
                    "author": c["author"],
                    "title": f"💬 Comment on: {post['title'][:60]}",
                    "selftext": c["body"],
                    "text": c["body"],
                    "score": c.get("score", 0),
                    "num_comments": 0,
                    "permalink": post["permalink"],
                    "created_utc": 0,
                    "is_comment": True,
                })
            
            # Rate limit between comment fetches
            time.sleep(3)
        
        comment_count += len(all_comment_items)
        
        # Combine posts + comments for analysis
        all_items_to_analyze = new_posts + all_comment_items
        
        add_event("status", {
            "message": f"⚡ วิเคราะห์ {len(new_posts)} โพสต์ + {len(all_comment_items)} คอมเมนต์... (รวม {len(all_results) + len(all_items_to_analyze)})",
            "phase": "analyzing",
        })
        
        batch_results, word_counter = predict_posts_parallel(
            all_items_to_analyze, word_counter, max_workers=4
        )
        
        if should_stop():
            break
        
        all_results.extend(batch_results)
        
        # Compute aggregates and emit progress
        agg = compute_aggregates(all_results, word_counter)
        
        add_event("progress", {
            "batch_results": batch_results,
            "total_processed": len(all_results),
            "total_scraped": len(seen_ids),
            "total_comments": comment_count,
            "round": round_num,
            "sort_mode": sort_label,
            **agg,
        })
        
        # Move to next sort mode
        sort_idx += 1
        
        # Rate limit between rounds
        time.sleep(3)
    
    # Stopped by user
    add_event("complete", {"message": f"หยุดการวิเคราะห์แล้ว — รวม {len(all_results)} รายการ ({len(seen_ids)} โพสต์ + {comment_count} คอมเมนต์)"})
    with jobs_lock:
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["completed"] = True


# ========== Flask Routes ==========
@app.route("/")
def index():
    return render_template("reddit_dashboard.html")


@app.route("/text")
def text_predictor():
    return render_template("text_predictor.html")


@app.route("/api/analyze", methods=["POST"])
def start_analysis():
    data = request.get_json()
    url_input = data.get("url", "").strip()
    
    subreddit = extract_subreddit_name(url_input)
    if not subreddit:
        return jsonify({"error": "กรุณาใส่ชื่อ subreddit หรือ URL ที่ถูกต้อง เช่น r/python หรือ https://reddit.com/r/python"}), 400
    
    job_id = str(uuid.uuid4())[:8]
    
    # Start background thread
    thread = threading.Thread(target=run_analysis_job, args=(job_id, subreddit), daemon=True)
    thread.start()
    
    return jsonify({"job_id": job_id, "subreddit": subreddit})


@app.route("/api/stop/<job_id>", methods=["POST"])
def stop_analysis(job_id):
    """Stop a running continuous analysis job."""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        job["stop"] = True
    return jsonify({"status": "stopping"})


@app.route("/api/stream/<job_id>")
def stream_results(job_id):
    """SSE endpoint for real-time result streaming."""
    def generate():
        last_idx = 0
        max_wait = 600  # 10 minutes max for continuous mode
        start_time = time.time()
        
        while True:
            if time.time() - start_time > max_wait:
                yield f"data: {json.dumps({'type': 'timeout', 'data': {'message': 'Timeout — 10 นาที'}})}\n\n"
                break
            
            with jobs_lock:
                job = jobs.get(job_id)
                if not job:
                    yield f"data: {json.dumps({'type': 'error', 'data': {'message': 'Job not found'}})}\n\n"
                    break
                
                events = job["events"][last_idx:]
                last_idx = len(job["events"])
                completed = job["completed"]
            
            for event in events:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            
            if completed:
                break
            
            time.sleep(0.5)
    
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


# ========== Text Predictor Routes (from mbti_project) ==========
@app.route("/api/types")
def api_types():
    """Return all 16 MBTI type profiles."""
    return jsonify(TYPE_PROFILES)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text or len(text.strip()) < 20:
        return jsonify({"error": "กรุณาใส่ข้อความอย่างน้อย 20 ตัวอักษร"}), 400

    # Preprocess
    cleaned = clean_text(text)

    # Vectorize
    text_vec = vectorizer.transform([cleaned])

    # Predict each dimension and collect probabilities
    result = {}
    mbti_type = ""
    dim_probas = {}

    for dim_key in ["EI", "NS", "TF", "JP"]:
        model = models[dim_key]
        raw_probas = model.predict_proba(text_vec)[0]
        classes = model.classes_

        # Apply bias correction
        corrected_probas = apply_bias_correction(dim_key, classes, raw_probas)

        dim_probas[dim_key] = {cls: corrected_probas[i] for i, cls in enumerate(classes)}

        # Pick prediction from corrected probabilities
        best_idx = corrected_probas.index(max(corrected_probas))
        prediction = classes[best_idx]

        dim_result = {}
        for i, cls in enumerate(classes):
            info = DIMENSION_INFO[dim_key][cls]
            dim_result[cls] = {
                "name": info["name"],
                "emoji": info["emoji"],
                "desc": info["desc"],
                "confidence": round(corrected_probas[i] * 100, 1),
            }

        result[dim_key] = {
            "prediction": prediction,
            "details": dim_result,
        }
        mbti_type += prediction

    nickname = TYPE_NICKNAMES.get(mbti_type, "Unknown Type")

    # Compute all 16 MBTI type probabilities
    ei_options = list(dim_probas["EI"].keys())
    ns_options = list(dim_probas["NS"].keys())
    tf_options = list(dim_probas["TF"].keys())
    jp_options = list(dim_probas["JP"].keys())

    all_types = []
    for e in ei_options:
        for n in ns_options:
            for t in tf_options:
                for j in jp_options:
                    code = e + n + t + j
                    prob = (
                        dim_probas["EI"][e]
                        * dim_probas["NS"][n]
                        * dim_probas["TF"][t]
                        * dim_probas["JP"][j]
                    )
                    all_types.append(
                        {
                            "type": code,
                            "nickname": TYPE_NICKNAMES.get(code, "Unknown"),
                            "probability": round(prob * 100, 2),
                        }
                    )

    all_types.sort(key=lambda x: x["probability"], reverse=True)
    alternatives = [t for t in all_types if t["type"] != mbti_type][:5]

    profile = TYPE_PROFILES.get(mbti_type, {})

    return jsonify(
        {
            "mbti_type": mbti_type,
            "nickname": nickname,
            "dimensions": result,
            "text_length": len(text),
            "words_count": len(text.split()),
            "alternatives": alternatives,
            "primary_probability": round(
                all_types[0]["probability"], 2
            ),
            "profile": profile,
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001, threaded=True)
