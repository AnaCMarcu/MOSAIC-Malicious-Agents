-- Table: users
CREATE TABLE users (
                    user_id TEXT PRIMARY KEY,
                    persona TEXT,
                    background_labels JSON,
                    creation_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    follower_count INTEGER DEFAULT 0,
                    total_likes_received INTEGER DEFAULT 0,
                    total_shares_received INTEGER DEFAULT 0,
                    total_comments_received INTEGER DEFAULT 0,
                    influence_score FLOAT DEFAULT 0.0,
                    is_influencer BOOLEAN DEFAULT FALSE,
                    last_influence_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

-- Table: posts
CREATE TABLE posts (
                    post_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    author_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    num_likes INTEGER DEFAULT 0,
                    num_shares INTEGER DEFAULT 0,
                    num_flags INTEGER DEFAULT 0,
                    num_comments INTEGER DEFAULT 0,
                    original_post_id TEXT,
                    is_news BOOLEAN DEFAULT FALSE,
                    news_type TEXT,
                    status TEXT CHECK(status IN ('active', 'taken_down')),
                    takedown_timestamp TIMESTAMP,
                    takedown_reason TEXT,
                    fact_check_status TEXT,
                    fact_checked_at TIMESTAMP,
                    FOREIGN KEY (author_id) REFERENCES users(user_id),
                    FOREIGN KEY (original_post_id) REFERENCES posts(post_id)
                );

-- Table: moderation_logs
CREATE TABLE moderation_logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id INTEGER,
                    action_type TEXT,
                    reason TEXT,
                    timestamp TIMESTAMP,
                    FOREIGN KEY (post_id) REFERENCES posts(post_id)
                );

-- Table: sqlite_sequence
CREATE TABLE sqlite_sequence(name,seq);

-- Table: community_notes
CREATE TABLE community_notes (
                    note_id TEXT PRIMARY KEY,
                    post_id TEXT NOT NULL,
                    author_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    helpful_ratings INTEGER DEFAULT 0,
                    not_helpful_ratings INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (post_id) REFERENCES posts(post_id),
                    FOREIGN KEY (author_id) REFERENCES users(user_id)
                );

-- Table: note_ratings
CREATE TABLE note_ratings (
                    note_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    rating TEXT CHECK(rating IN ('helpful', 'not_helpful')),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (note_id, user_id),
                    FOREIGN KEY (note_id) REFERENCES community_notes(note_id),
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );

-- Table: user_actions
CREATE TABLE user_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    target_id TEXT,
                    content TEXT,
                    reasoning TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );

-- Table: follows
CREATE TABLE follows (
                    follower_id TEXT NOT NULL,
                    followed_id TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (follower_id, followed_id),
                    FOREIGN KEY (follower_id) REFERENCES users(user_id),
                    FOREIGN KEY (followed_id) REFERENCES users(user_id)
                );

-- Table: comments
CREATE TABLE comments (
                    comment_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    post_id TEXT NOT NULL,
                    author_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    num_likes INTEGER DEFAULT 0,
                    FOREIGN KEY (post_id) REFERENCES posts(post_id),
                    FOREIGN KEY (author_id) REFERENCES users(user_id)
                );

-- Table: agent_memories
CREATE TABLE agent_memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance_score FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    decay_factor FLOAT DEFAULT 1.0,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );

-- Table: spread_metrics
CREATE TABLE spread_metrics (
                    post_id TEXT NOT NULL,
                    time_step INTEGER NOT NULL,
                    views INTEGER NOT NULL,
                    diffusion_depth INTEGER NOT NULL,
                    num_likes INTEGER NOT NULL,
                    num_shares INTEGER NOT NULL,
                    num_flags INTEGER NOT NULL,
                    num_comments INTEGER NOT NULL,
                    num_notes INTEGER NOT NULL,
                    num_note_ratings INTEGER NOT NULL,
                    total_interactions INTEGER NOT NULL,
                    should_takedown BOOLEAN,
                    takedown_reason TEXT,
                    takedown_executed BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (post_id, time_step),
                    FOREIGN KEY (post_id) REFERENCES posts(post_id)
                );

-- Table: feed_exposures
CREATE TABLE feed_exposures (
                    user_id TEXT NOT NULL,
                    post_id TEXT NOT NULL,
                    time_step INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, post_id, time_step),
                    FOREIGN KEY (user_id) REFERENCES users(user_id),
                    FOREIGN KEY (post_id) REFERENCES posts(post_id)
                );

-- Table: fact_checks
CREATE TABLE fact_checks (
                    post_id TEXT NOT NULL,
                    checker_id TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    explanation TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    sources TEXT NOT NULL,
                    groundtruth TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (post_id),
                    FOREIGN KEY (post_id) REFERENCES posts(post_id)
                );

