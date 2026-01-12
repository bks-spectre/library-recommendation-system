from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ===================== DATA GENERATION =====================

np.random.seed(42)

# Storage for feedbacks and ratings
feedbacks_list = []
book_ratings_list = []

genres = ['Fiction', 'Mystery', 'Science Fiction', 'Fantasy', 'Romance', 
          'History', 'Biography', 'Self-Help', 'Adventure', 'Thriller']

authors = ['Agatha Christie', 'George Orwell', 'J.K. Rowling', 'Paulo Coelho', 'Haruki Murakami',
           'Margaret Atwood', 'Stephen King', 'Jane Austen', 'Mark Twain', 'Charles Dickens',
           'Leo Tolstoy', 'Oscar Wilde', 'Arthur Conan Doyle', 'Isaac Asimov', 'Stephen Hawking']

# Realistic book titles by genre
book_titles = {
    'Fiction': ['The Silent Echo', 'Whispers in Time', 'The Last Garden', 'Beneath the Surface', 'Echoes of Tomorrow'],
    'Mystery': ['The Hidden Truth', 'Murder at Midnight', 'The Secret Witness', 'Dark Secrets', 'The Missing Piece'],
    'Science Fiction': ['Beyond the Stars', 'The Time Paradox', 'Galactic Warriors', 'The Future Awaits', 'Quantum Dreams'],
    'Fantasy': ['The Dragon\'s Legacy', 'Realm of Shadows', 'The Magic Stone', 'Sword of Destiny', 'The Enchanted Forest'],
    'Romance': ['Love in Paris', 'Hearts Entwined', 'The Last Kiss', 'Forever Yours', 'Summer of Love'],
    'History': ['The Ancient World', 'Empires Rise and Fall', 'War and Peace Chronicles', 'The Renaissance Era', 'Forgotten Kingdoms'],
    'Biography': ['Life of a Legend', 'The Journey Within', 'Against All Odds', 'Inspiring Lives', 'The Untold Story'],
    'Self-Help': ['Mindful Living', 'The Power Within', 'Habits of Success', 'Finding Your Purpose', 'The Better You'],
    'Adventure': ['Journey to the Unknown', 'The Explorer\'s Quest', 'Wild Horizons', 'Lost in the Jungle', 'Mountain Peak'],
    'Thriller': ['The Final Hour', 'Deadly Game', 'Run or Hide', 'The Chase', 'Edge of Danger']
}

# Indian student names
student_names = [
    'Aarav Sharma', 'Vivaan Patel', 'Aditya Kumar', 'Vihaan Singh', 'Arjun Gupta',
    'Sai Reddy', 'Arnav Verma', 'Ayaan Khan', 'Krishna Iyer', 'Ishaan Joshi',
    'Aadhya Sharma', 'Ananya Patel', 'Diya Kumar', 'Navya Singh', 'Ira Gupta',
    'Saanvi Reddy', 'Aanya Verma', 'Kiara Khan', 'Myra Iyer', 'Sara Joshi',
    'Rohan Mehta', 'Kabir Nair', 'Dhruv Desai', 'Advait Malhotra', 'Reyansh Pillai',
    'Aarohi Mehta', 'Pihu Nair', 'Riya Desai', 'Shanaya Malhotra', 'Tara Pillai',
    'Atharv Rao', 'Shaurya Chopra', 'Vedant Agarwal', 'Yash Bansal', 'Pranav Kapoor',
    'Anvi Rao', 'Avni Chopra', 'Pari Agarwal', 'Aaradhya Bansal', 'Nitya Kapoor',
    'Karthik Srinivasan', 'Amar Bose', 'Dev Kulkarni', 'Harsh Bhatt', 'Madhav Saxena',
    'Meera Srinivasan', 'Prisha Bose', 'Kavya Kulkarni', 'Zara Bhatt', 'Anika Saxena'
]

books_data = []
book_counter = {}

for i in range(1, 301):
    book_id = f"B{i:03d}"
    genre = np.random.choice(genres)
    
    if genre not in book_counter:
        book_counter[genre] = 0
    
    title_list = book_titles[genre]
    base_title = title_list[book_counter[genre] % len(title_list)]
    
    if book_counter[genre] >= len(title_list):
        title = f"{base_title} Vol. {(book_counter[genre] // len(title_list)) + 1}"
    else:
        title = base_title
    
    book_counter[genre] += 1
    
    author = np.random.choice(authors)
    pages = np.random.randint(150, 600)
    reading_level = np.random.choice(['Beginner', 'Intermediate', 'Advanced'])
    avg_rating = round(np.random.uniform(2.5, 5.0), 1)
    publication_year = np.random.randint(1980, 2024)
    
    books_data.append({
        'book_id': book_id,
        'title': title,
        'genre': genre,
        'author': author,
        'pages': pages,
        'reading_level': reading_level,
        'avg_rating': avg_rating,
        'publication_year': publication_year
    })

books_df = pd.DataFrame(books_data)

# Create student IDs and assign names
student_ids = [f"S{i:02d}" for i in range(1, 51)]
students_data = []

for idx, student_id in enumerate(student_ids):
    name = student_names[idx]
    grade = np.random.choice([11, 12])
    preference_genre = np.random.choice(genres)
    preferred_level = np.random.choice(['Beginner', 'Intermediate', 'Advanced'])
    books_read = np.random.randint(5, 20)
    
    students_data.append({
        'student_id': student_id,
        'name': name,
        'grade': grade,
        'preference_genre': preference_genre,
        'preferred_level': preferred_level,
        'books_read': books_read
    })

students_df = pd.DataFrame(students_data)

# Generate reading history
reading_history = []
for _, student in students_df.iterrows():
    num_books = student['books_read']
    sampled_books = books_df.sample(n=num_books)
    
    for _, book in sampled_books.iterrows():
        rating = np.random.randint(1, 6)
        date_read = f"2024-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}"
        
        reading_history.append({
            'student_id': student['student_id'],
            'book_id': book['book_id'],
            'rating': rating,
            'date_read': date_read
        })

reading_history_df = pd.DataFrame(reading_history)

# ===================== FEATURE ENGINEERING =====================

student_features = []

for _, student in students_df.iterrows():
    student_id = student['student_id']
    student_books = reading_history_df[reading_history_df['student_id'] == student_id]['book_id'].tolist()
    student_book_data = books_df[books_df['book_id'].isin(student_books)]
    
    avg_rating_given = reading_history_df[reading_history_df['student_id'] == student_id]['rating'].mean()
    genre_diversity = student_book_data['genre'].nunique()
    avg_pages_read = student_book_data['pages'].mean()
    
    genre_counts = student_book_data['genre'].value_counts().to_dict()
    favorite_genre_count = max(genre_counts.values()) if genre_counts else 0
    
    beginner_count = (student_book_data['reading_level'] == 'Beginner').sum()
    intermediate_count = (student_book_data['reading_level'] == 'Intermediate').sum()
    advanced_count = (student_book_data['reading_level'] == 'Advanced').sum()
    
    # New features for enhanced personalization
    avg_page_preference = avg_pages_read if len(student_book_data) > 0 else 300
    
    modern_books = (student_book_data['publication_year'] >= 1990).sum()
    classic_books = (student_book_data['publication_year'] < 1990).sum()
    modern_vs_classic_ratio = modern_books / (classic_books + 1) if len(student_book_data) > 0 else 1
    
    genre_diversity_score = genre_diversity if len(student_book_data) > 0 else 0
    
    student_features.append({
        'student_id': student_id,
        'avg_rating_given': avg_rating_given,
        'genre_diversity': genre_diversity,
        'avg_pages_read': avg_pages_read,
        'favorite_genre_count': favorite_genre_count,
        'beginner_count': beginner_count,
        'intermediate_count': intermediate_count,
        'advanced_count': advanced_count,
        'books_read': student['books_read'],
        'avg_page_preference': avg_page_preference,
        'modern_vs_classic_ratio': modern_vs_classic_ratio,
        'genre_diversity_score': genre_diversity_score
    })

student_features_df = pd.DataFrame(student_features)

# K-Means Clustering
scaler = StandardScaler()
features_to_scale = ['avg_rating_given', 'genre_diversity', 'avg_pages_read', 
                     'favorite_genre_count', 'beginner_count', 'intermediate_count', 
                     'advanced_count', 'books_read', 'avg_page_preference', 
                     'modern_vs_classic_ratio', 'genre_diversity_score']
scaled_features = scaler.fit_transform(student_features_df[features_to_scale])

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(scaled_features)

student_features_df['cluster'] = clusters

# ===================== COLLABORATIVE FILTERING =====================

# Create user-item matrix for collaborative filtering
user_item_matrix = reading_history_df.pivot_table(
    index='student_id',
    columns='book_id',
    values='rating',
    fill_value=0
)

# Calculate user similarity matrix
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)

# ===================== ASSOCIATION RULES =====================

def calculate_association_rules():
    transactions = []
    for _, student in students_df.iterrows():
        student_id = student['student_id']
        student_books = reading_history_df[reading_history_df['student_id'] == student_id]['book_id'].tolist()
        student_book_data = books_df[books_df['book_id'].isin(student_books)]
        
        transaction = list(student_book_data['genre'].unique())
        transactions.append(transaction)

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    transaction_df = pd.DataFrame(te_ary, columns=te.columns_)

    min_support = 0.12
    frequent_itemsets = apriori(transaction_df, min_support=min_support, use_colnames=True)

    association_rules_list = []
    if len(frequent_itemsets) > 0:
        min_confidence = 0.4
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        
        if len(rules) > 0:
            for idx, row in rules.iterrows():
                antecedent = ', '.join(list(row['antecedents']))
                consequent = ', '.join(list(row['consequents']))
                association_rules_list.append({
                    'from': antecedent,
                    'to': consequent,
                    'confidence': f"{row['confidence']:.1%}",
                    'support': f"{row['support']:.1%}",
                    'lift': f"{row['lift']:.2f}"
                })
    
    return association_rules_list

# ===================== RECOMMENDATION FUNCTIONS =====================

def get_collaborative_recommendations(student_id, n_recommendations=3):
    """Get recommendations using collaborative filtering"""
    if student_id not in user_similarity_df.index:
        return []
    
    # Find similar users
    similar_users = user_similarity_df[student_id].sort_values(ascending=False)[1:6]
    
    # Get books read by student
    read_books = set(reading_history_df[reading_history_df['student_id'] == student_id]['book_id'])
    
    # Get books rated highly by similar users
    recommendations = {}
    for similar_user_id, similarity_score in similar_users.items():
        similar_user_books = reading_history_df[
            (reading_history_df['student_id'] == similar_user_id) & 
            (reading_history_df['rating'] >= 4)
        ]
        
        for _, row in similar_user_books.iterrows():
            book_id = row['book_id']
            if book_id not in read_books:
                if book_id not in recommendations:
                    recommendations[book_id] = 0
                recommendations[book_id] += similarity_score * row['rating']
    
    # Sort and return top recommendations
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    recommended_book_ids = [book_id for book_id, score in sorted_recommendations[:n_recommendations]]
    
    return books_df[books_df['book_id'].isin(recommended_book_ids)].to_dict('records')


def get_hybrid_recommendations(student_id, num_recommendations=5, reading_level_filter=None):
    """Hybrid recommendation combining multiple approaches"""
    if student_id not in students_df['student_id'].values:
        return []
    
    student = students_df[students_df['student_id'] == student_id].iloc[0]
    student_cluster = student_features_df[student_features_df['student_id'] == student_id]['cluster'].iloc[0]
    
    read_books = set(reading_history_df[reading_history_df['student_id'] == student_id]['book_id'].tolist())
    
    if reading_level_filter:
        available_books = books_df[books_df['reading_level'] == reading_level_filter]
    else:
        available_books = books_df.copy()
    
    available_books = available_books[~available_books['book_id'].isin(read_books)]
    
    if len(available_books) == 0:
        return []
    
    # Get collaborative filtering recommendations
    collab_recs = get_collaborative_recommendations(student_id, n_recommendations=10)
    collab_book_ids = [book['book_id'] for book in collab_recs]
    
    scores = []
    for _, book in available_books.iterrows():
        score = 0
        
        # Content-based: Genre preference (supports multiple genres)
        if 'favorite_genres' in student and isinstance(student['favorite_genres'], list):
            if book['genre'] in student['favorite_genres']:
                score += 30
        elif 'preference_genre' in student:
            if book['genre'] == student['preference_genre']:
                score += 30
        
        # Content-based: Book rating
        score += book['avg_rating'] * 10
        
        # Page length preference
        if 'page_length' in student:
            page_pref = student['page_length']
            if page_pref == 'Short' and book['pages'] < 250:
                score += 20
            elif page_pref == 'Medium' and 250 <= book['pages'] <= 400:
                score += 20
            elif page_pref == 'Long' and book['pages'] > 400:
                score += 20
        
        # Publication period preference
        if 'publication_period' in student:
            period_pref = student['publication_period']
            if period_pref == 'Classic' and book['publication_year'] < 1990:
                score += 15
            elif period_pref == 'Modern' and 1990 <= book['publication_year'] <= 2010:
                score += 15
            elif period_pref == 'Contemporary' and book['publication_year'] > 2010:
                score += 15
        
        # Reading goal preference
        if 'reading_goal' in student:
            goal = student['reading_goal']
            if goal == 'Learning & Growth' and book['reading_level'] == 'Advanced':
                score += 12
            elif goal == 'Entertainment' and book['genre'] in ['Fiction', 'Adventure', 'Thriller']:
                score += 12
            elif goal == 'Escape & Relaxation' and book['genre'] in ['Fantasy', 'Romance']:
                score += 12
            elif goal == 'Expanding Knowledge' and book['genre'] in ['History', 'Biography', 'Self-Help']:
                score += 12
        
        # Cluster-based: Books popular in same cluster
        cluster_students = student_features_df[student_features_df['cluster'] == student_cluster]['student_id'].tolist()
        cluster_read_ratings = reading_history_df[
            (reading_history_df['student_id'].isin(cluster_students)) & 
            (reading_history_df['book_id'] == book['book_id'])
        ]['rating'].tolist()
        
        if cluster_read_ratings:
            score += np.mean(cluster_read_ratings) * 8
        
        # Collaborative filtering boost
        if book['book_id'] in collab_book_ids:
            score += 25
        
        # Reading level match
        if book['reading_level'] == student['preferred_level']:
            score += 15
        
        # Recency bonus (newer books)
        if book['publication_year'] >= 2020:
            score += 10
        elif book['publication_year'] >= 2010:
            score += 5
        
        scores.append(score)
    
    available_books_copy = available_books.copy()
    available_books_copy['score'] = scores
    recommended_books = available_books_copy.nlargest(num_recommendations, 'score')
    
    return recommended_books.to_dict('records')


def get_student_insights(student_id):
    """Get reading insights for a student"""
    
    cluster_names = {
        0: "Casual Readers",
        1: "Diverse Explorers",
        2: "Genre Enthusiasts",
        3: "Advanced Scholars",
        4: "Avid Bookworms"
    }
    
    if student_id not in students_df['student_id'].values:
        return None
    
    student = students_df[students_df['student_id'] == student_id].iloc[0]
    student_history = reading_history_df[reading_history_df['student_id'] == student_id]
    student_books = books_df[books_df['book_id'].isin(student_history['book_id'])]
    
    # Calculate insights
    avg_rating = student_history['rating'].mean()
    total_books = len(student_history)
    total_pages = student_books['pages'].sum()
    
    # Favorite genre
    genre_counts = student_books['genre'].value_counts()
    favorite_genre = genre_counts.index[0] if len(genre_counts) > 0 else 'None'
    
    # Reading level distribution
    level_dist = student_books['reading_level'].value_counts().to_dict()
    
    # Get cluster info
    cluster = student_features_df[student_features_df['student_id'] == student_id]['cluster'].iloc[0]
    cluster_size = (student_features_df['cluster'] == cluster).sum()
    
    # Construct preference genre info
    preference_genre = student.get('preference_genre', 'Not specified')
    if 'favorite_genres' in student and isinstance(student['favorite_genres'], list):
        preference_genre = ', '.join(student['favorite_genres'])
    
    return {
        'student_name': str(student['name']),
        'grade': int(student['grade']),
        'total_books': int(total_books),
        'total_pages': int(total_pages),
        'avg_rating': float(round(avg_rating, 1)),
        'favorite_genre': str(favorite_genre),
        'level_distribution': {str(k): int(v) for k, v in level_dist.items()},
        'cluster': int(cluster),
        'cluster_name': cluster_names[int(cluster)],
        'cluster_size': int(cluster_size),
        'preference_genre': str(preference_genre)
    }

    

# ===================== ROUTES =====================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📚 AI Library Recommendation System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        /* Hamburger Menu */
        .hamburger {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            background: white;
            border: none;
            border-radius: 8px;
            padding: 12px;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            transition: transform 0.3s;
        }

        .hamburger:hover {
            transform: scale(1.05);
        }

        .hamburger span {
            display: block;
            width: 25px;
            height: 3px;
            background: #667eea;
            margin: 5px 0;
            transition: 0.3s;
        }

        .nav-menu {
            position: fixed;
            top: 0;
            right: -300px;
            width: 300px;
            height: 100vh;
            background: white;
            box-shadow: -5px 0 15px rgba(0,0,0,0.3);
            z-index: 999;
            transition: right 0.3s ease-in-out;
            will-change: transform;
            transform: translateZ(0);
            padding: 80px 30px 30px;
            overflow-y: auto;
        }

        .nav-menu.active {
            right: 0;
        }

        .nav-menu a {
            display: block;
            padding: 15px 0;
            color: #333;
            text-decoration: none;
            font-weight: 500;
            border-bottom: 1px solid #eee;
            transition: 0.3s;
        }

        .nav-menu a:hover {
            color: #667eea;
            padding-left: 10px;
        }

        header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }

        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        header p {
            font-size: 1.1em;
            opacity: 0.9;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .card {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s, box-shadow 0.3s;
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        }

        .stat-card h3 {
            color: #667eea;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
            opacity: 0.8;
        }

        .stat-card .value {
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
        }

        .section-title {
            color: #333;
            font-size: 1.5em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
            display: inline-block;
        }

        .recommendations-container {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }

        .input-group {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            position: relative;
        }

        .search-wrapper {
            position: relative;
            flex: 1;
            min-width: 200px;
        }

        .input-group input[type="text"] {
            width: 100%;
            padding: 10px 15px;
            border: 2px solid #667eea;
            border-radius: 8px;
            font-size: 1em;
        }

        .input-group input[type="text"]:focus {
            outline: none;
            box-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
        }

        .search-results {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: white;
            border: 2px solid #667eea;
            border-top: none;
            border-radius: 0 0 8px 8px;
            max-height: 200px;
            overflow-y: auto;
            z-index: 100;
            display: none;
        }

        .search-results.active {
            display: block;
        }

        .search-item {
            padding: 10px 15px;
            cursor: pointer;
            transition: background 0.2s;
        }

        .search-item:hover {
            background: #f0f0f0;
        }

        .input-group select, .input-group button {
            padding: 10px 15px;
            border: 2px solid #667eea;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
            transition: 0.3s;
        }

        .input-group select {
            background: white;
            color: #333;
            flex: 1;
            min-width: 200px;
        }

        .input-group button {
            background: #667eea;
            color: white;
            border: none;
            font-weight: bold;
            min-width: 150px;
        }

        .input-group button:hover {
            background: #764ba2;
            transform: scale(1.02);
        }

        .btn-secondary {
            background: #6c757d !important;
        }

        .btn-secondary:hover {
            background: #5a6268 !important;
        }

        .book-item {
            background: #f8f9fa;
            padding: 15px;
            margin-bottom: 12px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            transition: 0.3s;
        }

        .book-item:hover {
            background: #e9ecef;
            transform: translateX(5px);
        }

        .book-title {
            font-weight: bold;
            color: #333;
            font-size: 1.1em;
            margin-bottom: 5px;
        }

        .book-meta {
            font-size: 0.9em;
            color: #666;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 8px;
        }

        .rating {
            color: #f39c12;
            font-weight: bold;
        }

        .genre-badge {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.85em;
        }

        .insights-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }

        .insight-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }

        .insight-label {
            color: #666;
            font-size: 0.9em;
            margin-bottom: 5px;
        }

        .insight-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }

        .clustering-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }

        .cluster-item {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }

        .cluster-item .label {
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 5px;
        }

        .cluster-item .count {
            font-size: 1.8em;
            font-weight: bold;
        }

        .genre-dist {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }

        .genre-bar {
            margin-bottom: 15px;
        }

        .genre-bar-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-weight: bold;
            color: #333;
        }

        .genre-bar-fill {
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 30px;
            border-radius: 5px;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            color: white;
            font-weight: bold;
        }

        .modal {
            display: none;
            position: fixed;
            z-index: 1001;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            animation: fadeIn 0.3s;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .modal.active {
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .modal-content {
    background: white;
    border-radius: 12px;
    padding: 30px;
    max-width: 1200px;
    max-height: 80vh;
    overflow-y: auto;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
}

#booksModal .modal-content,
#genreModal .modal-content {
    max-width: 90vw;
    width: 1200px;
}

#booksContent,
#genreContent {
    min-width: 100%;
}

.genre-bar {
    min-width: 600px;
}

        .modal-close {
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
            color: #999;
        }

        .star-rating, .book-star-rating {
            cursor: pointer;
            transition: color 0.2s;
            user-select: none;
        }

        .star-rating:hover, .book-star-rating:hover {
            color: #f39c12;
        }

        .star-rating.active, .book-star-rating.active {
            color: #f39c12;
        }

        .modal-close:hover {
            color: #333;
        }

        .rule-item {
            background: #f8f9fa;
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }

        .rule-item .arrow {
            color: #667eea;
            font-weight: bold;
        }

        .confidence {
            background: #667eea;
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }

        .loading {
            text-align: center;
            padding: 20px;
            color: #667eea;
        }

        .no-results {
            text-align: center;
            padding: 30px;
            color: #999;
            background: #f8f9fa;
            border-radius: 8px;
        }

        footer {
            text-align: center;
            color: white;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid rgba(255,255,255,0.2);
            opacity: 0.8;
        }

        @media (max-width: 768px) {
            header h1 {
                font-size: 1.8em;
            }

            .grid {
                grid-template-columns: 1fr;
            }

            .input-group {
                flex-direction: column;
            }

            .input-group select, .input-group button, .search-wrapper {
                width: 100%;
            }
        }
    </style>
</head>
<body>

<button class="hamburger" onclick="toggleMenu()">
        <span></span>
        <span></span>
        <span></span>
    </button>

    <nav class="nav-menu" id="navMenu">
    <a href="#home">🏠 Home</a>
    <a href="javascript:void(0)" onclick="showRegistrationForm()">➕ Register New Student</a>
    <a href="javascript:void(0)" onclick="showAllBooks()">📖 Browse All Books</a>
    <a href="#recommendations">🎯 Recommendations</a>
    <a href="#insights">📊 Student Insights</a>
    <a href="#clustering">🔍 Clustering</a>
    <a href="javascript:void(0)" onclick="showGenreDistribution()">📚 Genre Distribution</a>
    <a href="javascript:void(0)" onclick="showAssociationRules()">🔗 Association Rules</a>
    <a href="javascript:void(0)" onclick="showFeedbackForm()">💬 Give Feedback</a>
    <a href="javascript:void(0)" onclick="showBookRating()">⭐ Rate a Book</a>
</nav>

    <div class="container">
        <header id="home">
            <h1>📚 AI Library Recommendation System</h1>
            <p>Hybrid AI-Powered Personalized Book Suggestions for Students</p>
        </header>

        <div class="grid">
            <div class="card stat-card">
                <h3>📖 Total Books</h3>
                <div class="value">{{ stats.total_books }}</div>
            </div>
            <div class="card stat-card">
                <h3>👥 Total Students</h3>
                <div class="value">{{ stats.total_students }}</div>
            </div>
            <div class="card stat-card">
                <h3>⭐ Avg Rating</h3>
                <div class="value">{{ stats.avg_rating }}/5</div>
            </div>
            <div class="card stat-card">
                <h3>📊 Reading Records</h3>
                <div class="value">{{ stats.total_records }}</div>
            </div>
        </div>

        <div class="recommendations-container" id="recommendations">
            <h2 class="section-title">🎯 Get Hybrid Recommendations</h2>
            <p style="color: #666; margin-bottom: 15px;">Uses K-Means Clustering, Collaborative Filtering & Content-Based Filtering</p>
            <div class="input-group">
                <div class="search-wrapper">
                    <input type="text" id="studentSearch" placeholder="Search student by name..." 
                           onkeyup="searchStudents()" onfocus="searchStudents()">
                    <div class="search-results" id="studentSearchResults"></div>
                </div>
                <select id="levelFilter">
                    <option value="">All Reading Levels</option>
                    <option value="Beginner">Beginner</option>
                    <option value="Intermediate">Intermediate</option>
                    <option value="Advanced">Advanced</option>
                </select>
                <button onclick="getRecommendations()">Get Recommendations</button>
            </div>
            <div id="recommendationsResult"></div>
        </div>

        <div class="recommendations-container" id="insights">
            <h2 class="section-title">📊 Student Reading Insights</h2>
            <div class="input-group">
                <div class="search-wrapper">
                    <input type="text" id="insightSearch" placeholder="Search student for insights..." 
                           onkeyup="searchStudentsForInsights()" onfocus="searchStudentsForInsights()">
                    <div class="search-results" id="insightSearchResults"></div>
                </div>
                <button onclick="getInsights()">View Insights</button>
            </div>
            <div id="insightsResult"></div>
        </div>

        <div class="card" id="clustering">
            <h2 class="section-title">🔍 Student Clustering Analysis</h2>
            <p style="color: #666; margin-bottom: 15px;">Students grouped into {{ cluster_stats|length }} clusters using K-Means based on reading behavior:</p>
            <div class="clustering-grid">
                {% for cluster in cluster_stats %}
                <div class="cluster-item">
                    <div class="label">{{ cluster.name }}</div>
                    <div class="count">{{ cluster.count }}</div>
                    <div class="label" style="font-size: 0.8em;">students</div>
                </div>
                {% endfor %}
            </div>
        </div>

        <footer>
            <p>🎓 CBSE Class 12 AI Project | Advanced Library Recommendation System</p>
            <p style="font-size: 0.9em; margin-top: 10px;">
                Using: K-Means Clustering, Collaborative Filtering, Content-Based Filtering, Apriori Algorithm
            </p>
        </footer>
    </div>

    <!-- Association Rules Modal -->
    <div id="rulesModal" class="modal">
        <div class="modal-content">
            <span class="modal-close" onclick="closeModal()">&times;</span>
            <h2 style="color: #667eea; margin-bottom: 20px;">🔗 Genre Association Rules</h2>
            <p style="color: #666; margin-bottom: 20px;">Discovered using Apriori Algorithm - Shows frequently read genre combinations:</p>
            <div id="rulesContent">
                <div class="loading">Loading association rules...</div>
            </div>
        </div>
    </div>

<!-- Genre Distribution Modal -->
<div id="genreModal" class="modal">
    <div class="modal-content">
        <span class="modal-close" onclick="closeGenreModal()">&times;</span>
        <h2 style="color: #667eea; margin-bottom: 20px;">📚 Genre Distribution</h2>
        <div id="genreContent">
            {% for genre in genre_distribution %}
            <div class="genre-bar">
                <div class="genre-bar-label">
                    <span>{{ genre.name }}</span>
                    <span>{{ genre.count }} books</span>
                </div>
                <div class="genre-bar-fill" style="width: {{ genre.percentage }}%">
                    {{ genre.percentage }}%
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
</div>

<!-- Browse Books Modal -->
<div id="booksModal" class="modal">
    <div class="modal-content">
        <span class="modal-close" onclick="closeBooksModal()">&times;</span>
        <h2 style="color: #667eea; margin-bottom: 20px;">📖 Browse All Books</h2>
        
        <!-- Add search and filter -->
        <div style="margin-bottom: 20px;">
            <input type="text" id="bookSearchInput" placeholder="Search by title, author, or genre..." 
                   style="width: 100%; padding: 10px; border: 2px solid #667eea; border-radius: 8px;"
                   onkeyup="filterBooks()">
        </div>
        
        <div id="booksContent" style="max-height: 500px; overflow-y: auto;">
            <div class="loading">Loading books...</div>
        </div>
    </div>
</div>

<!-- Student Registration Modal -->
<div id="registrationModal" class="modal">
    <div class="modal-content">
        <span class="modal-close" onclick="closeRegistrationModal()">&times;</span>
        <h2 style="color: #667eea; margin-bottom: 20px;">➕ Register New Student</h2>
        <p style="color: #666; margin-bottom: 20px;">Fill in your details to get personalized book recommendations!</p>
        
        <form id="registrationForm" style="display: grid; gap: 15px;">
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: bold; color: #333;">Full Name *</label>
                <input type="text" id="regName" required
                       style="width: 100%; padding: 10px; border: 2px solid #667eea; border-radius: 8px; font-size: 1em;">
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div>
                    <label style="display: block; margin-bottom: 5px; font-weight: bold; color: #333;">Grade *</label>
                    <select id="regGrade" required
                            style="width: 100%; padding: 10px; border: 2px solid #667eea; border-radius: 8px; font-size: 1em;">
                        <option value="">Select Grade</option>
                        <option value="11">Grade 11</option>
                        <option value="12">Grade 12</option>
                    </select>
                </div>
                
                <div>
                    <label style="display: block; margin-bottom: 5px; font-weight: bold; color: #333;">Reading Level *</label>
                    <select id="regLevel" required
                            style="width: 100%; padding: 10px; border: 2px solid #667eea; border-radius: 8px; font-size: 1em;">
                        <option value="">Select Level</option>
                        <option value="Beginner">Beginner</option>
                        <option value="Intermediate">Intermediate</option>
                        <option value="Advanced">Advanced</option>
                    </select>
                </div>
            </div>
            
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: bold; color: #333;">Favorite Genres (Select 1-3) *</label>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 10px;">
                    <label style="display: flex; align-items: center; font-weight: normal; color: #333;">
                        <input type="checkbox" value="Fiction" class="genreCheckbox" style="margin-right: 8px;"> Fiction
                    </label>
                    <label style="display: flex; align-items: center; font-weight: normal; color: #333;">
                        <input type="checkbox" value="Mystery" class="genreCheckbox" style="margin-right: 8px;"> Mystery
                    </label>
                    <label style="display: flex; align-items: center; font-weight: normal; color: #333;">
                        <input type="checkbox" value="Science Fiction" class="genreCheckbox" style="margin-right: 8px;"> Science Fiction
                    </label>
                    <label style="display: flex; align-items: center; font-weight: normal; color: #333;">
                        <input type="checkbox" value="Fantasy" class="genreCheckbox" style="margin-right: 8px;"> Fantasy
                    </label>
                    <label style="display: flex; align-items: center; font-weight: normal; color: #333;">
                        <input type="checkbox" value="Romance" class="genreCheckbox" style="margin-right: 8px;"> Romance
                    </label>
                    <label style="display: flex; align-items: center; font-weight: normal; color: #333;">
                        <input type="checkbox" value="History" class="genreCheckbox" style="margin-right: 8px;"> History
                    </label>
                    <label style="display: flex; align-items: center; font-weight: normal; color: #333;">
                        <input type="checkbox" value="Biography" class="genreCheckbox" style="margin-right: 8px;"> Biography
                    </label>
                    <label style="display: flex; align-items: center; font-weight: normal; color: #333;">
                        <input type="checkbox" value="Self-Help" class="genreCheckbox" style="margin-right: 8px;"> Self-Help
                    </label>
                    <label style="display: flex; align-items: center; font-weight: normal; color: #333;">
                        <input type="checkbox" value="Adventure" class="genreCheckbox" style="margin-right: 8px;"> Adventure
                    </label>
                    <label style="display: flex; align-items: center; font-weight: normal; color: #333;">
                        <input type="checkbox" value="Thriller" class="genreCheckbox" style="margin-right: 8px;"> Thriller
                    </label>
                </div>
                <div id="genreError" style="color: #dc3545; font-size: 0.9em; display: none;">Please select 1-3 genres</div>
            </div>
            
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: bold; color: #333;">Preferred Page Length *</label>
                <select id="regPageLength" required
                        style="width: 100%; padding: 10px; border: 2px solid #667eea; border-radius: 8px; font-size: 1em;">
                    <option value="">Select Preferred Length</option>
                    <option value="Short">Short (< 250 pages)</option>
                    <option value="Medium">Medium (250-400 pages)</option>
                    <option value="Long">Long (> 400 pages)</option>
                </select>
            </div>
            
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: bold; color: #333;">Book Era Preference *</label>
                <select id="regPublicationPeriod" required
                        style="width: 100%; padding: 10px; border: 2px solid #667eea; border-radius: 8px; font-size: 1em;">
                    <option value="">Select Book Era</option>
                    <option value="Classic">Classic (Before 1990)</option>
                    <option value="Modern">Modern (1990-2010)</option>
                    <option value="Contemporary">Contemporary (2010-Present)</option>
                </select>
            </div>
            
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: bold; color: #333;">Reading Purpose *</label>
                <select id="regReadingGoal" required
                        style="width: 100%; padding: 10px; border: 2px solid #667eea; border-radius: 8px; font-size: 1em;">
                    <option value="">Select Your Purpose</option>
                    <option value="Entertainment">Entertainment</option>
                    <option value="Learning & Growth">Learning & Growth</option>
                    <option value="Escape & Relaxation">Escape & Relaxation</option>
                    <option value="Expanding Knowledge">Expanding Knowledge</option>
                    <option value="All of the above">All of the above</option>
                </select>
            </div>
            
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: bold; color: #333;">Genre Exploration Preference *</label>
                <select id="regGenreDiversity" required
                        style="width: 100%; padding: 10px; border: 2px solid #667eea; border-radius: 8px; font-size: 1em;">
                    <option value="">Select Your Preference</option>
                    <option value="Stick to one genre">Stick to one genre</option>
                    <option value="Explore 2-3 genres">Explore 2-3 genres</option>
                    <option value="Try many different genres">Try many different genres</option>
                </select>
            </div>
            
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: bold; color: #333;">How many books have you read in the past year? *</label>
                <input type="number" id="regBooksRead" min="0" max="100" required
                       style="width: 100%; padding: 10px; border: 2px solid #667eea; border-radius: 8px; font-size: 1em;">
            </div>
            
            <div id="registrationMessage" style="display: none; padding: 15px; border-radius: 8px; margin-top: 10px;"></div>
            
            <button type="submit" onclick="registerStudent(event)"
                    style="background: #667eea; color: white; padding: 15px; border: none; border-radius: 8px; 
                           font-size: 1.1em; font-weight: bold; cursor: pointer; transition: 0.3s;">
                Register & Get Recommendations
            </button>
        </form>
    </div>
</div>

<!-- Feedback Modal -->
<div id="feedbackModal" class="modal">
    <div class="modal-content">
        <span class="modal-close" onclick="closeFeedbackModal()">&times;</span>
        <h2 style="color: #667eea; margin-bottom: 20px;">💬 Give Feedback</h2>
        <p style="color: #666; margin-bottom: 20px;">Help us improve the recommendation system!</p>
        
        <form id="feedbackForm" style="display: grid; gap: 15px;">
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: bold; color: #333;">Your Name *</label>
                <input type="text" id="feedbackName" required
                       style="width: 100%; padding: 10px; border: 2px solid #667eea; border-radius: 8px; font-size: 1em;">
            </div>
            
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: bold; color: #333;">Student ID (Optional)</label>
                <input type="text" id="feedbackStudentId"
                       style="width: 100%; padding: 10px; border: 2px solid #667eea; border-radius: 8px; font-size: 1em;"
                       placeholder="e.g., S01">
            </div>
            
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: bold; color: #333;">Rating *</label>
                <div style="display: flex; gap: 10px; font-size: 2em;">
                    <span class="star-rating" data-rating="1" onclick="setFeedbackRating(1)">☆</span>
                    <span class="star-rating" data-rating="2" onclick="setFeedbackRating(2)">☆</span>
                    <span class="star-rating" data-rating="3" onclick="setFeedbackRating(3)">☆</span>
                    <span class="star-rating" data-rating="4" onclick="setFeedbackRating(4)">☆</span>
                    <span class="star-rating" data-rating="5" onclick="setFeedbackRating(5)">☆</span>
                </div>
                <input type="hidden" id="feedbackRating" value="0">
            </div>
            
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: bold; color: #333;">Your Feedback *</label>
                <textarea id="feedbackText" rows="5" required
                          style="width: 100%; padding: 10px; border: 2px solid #667eea; border-radius: 8px; font-size: 1em; resize: vertical;"></textarea>
            </div>
            
            <div id="feedbackMessage" style="display: none; padding: 15px; border-radius: 8px;"></div>
            
            <button type="submit" onclick="submitFeedback(event)"
                    style="background: #667eea; color: white; padding: 15px; border: none; border-radius: 8px; 
                           font-size: 1.1em; font-weight: bold; cursor: pointer; transition: 0.3s;">
                Submit Feedback
            </button>
        </form>
    </div>
</div>

<!-- Book Rating Modal -->
<div id="bookRatingModal" class="modal">
    <div class="modal-content">
        <span class="modal-close" onclick="closeBookRatingModal()">&times;</span>
        <h2 style="color: #667eea; margin-bottom: 20px;">⭐ Rate a Book</h2>
        <p style="color: #666; margin-bottom: 20px;">Share your reading experience!</p>
        
        <form id="bookRatingForm" style="display: grid; gap: 15px;">
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: bold; color: #333;">Your Student ID *</label>
                <div class="search-wrapper">
                    <input type="text" id="ratingStudentSearch" placeholder="Search your ID or name..." 
                           onkeyup="searchStudentsForRating()" onfocus="searchStudentsForRating()"
                           style="width: 100%; padding: 10px; border: 2px solid #667eea; border-radius: 8px; font-size: 1em;">
                    <div class="search-results" id="ratingStudentSearchResults"></div>
                </div>
            </div>
            
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: bold; color: #333;">Search Book *</label>
                <div class="search-wrapper">
                    <input type="text" id="ratingBookSearch" placeholder="Search book by title..." 
                           onkeyup="searchBooksForRating()" onfocus="searchBooksForRating()"
                           style="width: 100%; padding: 10px 15px; border: 2px solid #667eea; border-radius: 8px; font-size: 1em;">
                    <div class="search-results" id="ratingBookSearchResults"></div>
                </div>
            </div>
            
            <div id="existingRatingNotice" style="display: none; background: #fff3cd; border: 2px solid #ffc107; padding: 15px; border-radius: 8px;">
                <p style="color: #856404; margin-bottom: 10px;"><strong>You've already rated this book!</strong></p>
                <p style="color: #856404; margin-bottom: 10px;">Your current rating: <span id="currentRatingDisplay"></span></p>
                <button type="button" onclick="enableRatingEdit()"
                        style="background: #ffc107; color: #000; padding: 8px 15px; border: none; border-radius: 5px; cursor: pointer;">
                    Edit Rating
                </button>
            </div>
            
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: bold; color: #333;">Your Rating *</label>
                <div style="display: flex; gap: 10px; font-size: 2em;">
                    <span class="book-star-rating" data-rating="1" onclick="setBookRating(1)">☆</span>
                    <span class="book-star-rating" data-rating="2" onclick="setBookRating(2)">☆</span>
                    <span class="book-star-rating" data-rating="3" onclick="setBookRating(3)">☆</span>
                    <span class="book-star-rating" data-rating="4" onclick="setBookRating(4)">☆</span>
                    <span class="book-star-rating" data-rating="5" onclick="setBookRating(5)">☆</span>
                </div>
                <input type="hidden" id="bookRatingValue" value="0">
            </div>
            
            <div>
                <label style="display: block; margin-bottom: 5px; font-weight: bold; color: #333;">Review (Optional)</label>
                <textarea id="bookReview" rows="4"
                          style="width: 100%; padding: 10px; border: 2px solid #667eea; border-radius: 8px; font-size: 1em; resize: vertical;"
                          placeholder="Share your thoughts about this book..."></textarea>
            </div>
            
            <div id="bookRatingMessage" style="display: none; padding: 15px; border-radius: 8px;"></div>
            
            <button type="submit" onclick="submitBookRating(event)" id="submitRatingBtn"
                    style="background: #667eea; color: white; padding: 15px; border: none; border-radius: 8px; 
                           font-size: 1.1em; font-weight: bold; cursor: pointer; transition: 0.3s;">
                Submit Rating
            </button>
        </form>
    </div>
</div>

    <script>
        let students = {{ students|tojson }};
        let selectedStudentId = '';
        let selectedInsightStudentId = '';

        function toggleMenu() {
            const menu = document.getElementById('navMenu');
            menu.classList.toggle('active');
        }

function showAllBooks() {
    document.getElementById('booksModal').classList.add('active');
    document.getElementById('navMenu').classList.remove('active');
    loadAllBooks();
}

function closeBooksModal() {
    document.getElementById('booksModal').classList.remove('active');
}

function loadAllBooks() {
    fetch('/get_all_books')
        .then(response => response.json())
        .then(data => {
            displayBooks(data.books);
        })
        .catch(error => {
            document.getElementById('booksContent').innerHTML = 
                '<div class="no-results">Error loading books.</div>';
        });
}

function displayBooks(books) {
    let html = '';
    books.forEach(book => {
        html += `
            <div class="book-item">
                <div class="book-title">${book.title}</div>
                <div class="book-meta">
                    <span><strong>Author:</strong> ${book.author}</span>
                    <span><strong>Genre:</strong> <span class="genre-badge">${book.genre}</span></span>
                    <span><strong>Rating:</strong> <span class="rating">⭐ ${book.avg_rating}/5</span></span>
                    <span><strong>Pages:</strong> ${book.pages}</span>
                    <span><strong>Level:</strong> ${book.reading_level}</span>
                    <span><strong>Year:</strong> ${book.publication_year}</span>
                </div>
            </div>
        `;
    });
    document.getElementById('booksContent').innerHTML = html;
}

function filterBooks() {
    const searchTerm = document.getElementById('bookSearchInput').value.toLowerCase();
    
    fetch('/get_all_books')
        .then(response => response.json())
        .then(data => {
            const filtered = data.books.filter(book => 
                book.title.toLowerCase().includes(searchTerm) ||
                book.author.toLowerCase().includes(searchTerm) ||
                book.genre.toLowerCase().includes(searchTerm)
            );
            displayBooks(filtered);
        });
}

function showRegistrationForm() {
    document.getElementById('registrationModal').classList.add('active');
    document.getElementById('navMenu').classList.remove('active');
}

function closeRegistrationModal() {
    document.getElementById('registrationModal').classList.remove('active');
    document.getElementById('registrationForm').reset();
    document.getElementById('registrationMessage').style.display = 'none';
    document.getElementById('genreError').style.display = 'none';
    document.querySelectorAll('.genreCheckbox').forEach(cb => cb.checked = false);
}

function registerStudent(event) {
    event.preventDefault();
    
    const name = document.getElementById('regName').value;
    const grade = document.getElementById('regGrade').value;
    const level = document.getElementById('regLevel').value;
    const booksRead = document.getElementById('regBooksRead').value;
    const pageLength = document.getElementById('regPageLength').value;
    const publicationPeriod = document.getElementById('regPublicationPeriod').value;
    const readingGoal = document.getElementById('regReadingGoal').value;
    const genreDiversity = document.getElementById('regGenreDiversity').value;
    
    // Get selected genres
    const genreCheckboxes = document.querySelectorAll('.genreCheckbox:checked');
    const selectedGenres = Array.from(genreCheckboxes).map(cb => cb.value);
    
    // Validate genres (1-3 required)
    if (selectedGenres.length < 1 || selectedGenres.length > 3) {
        document.getElementById('genreError').style.display = 'block';
        return;
    } else {
        document.getElementById('genreError').style.display = 'none';
    }
    
    if (!name || !grade || !level || !booksRead || !pageLength || !publicationPeriod || !readingGoal || !genreDiversity) {
        showRegistrationMessage('Please fill in all required fields!', 'error');
        return;
    }
    
    // Disable the submit button to prevent double submission
    const form = document.getElementById('registrationForm');
    const submitBtn = form.querySelector('button[type="submit"]');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Registering...';
    
    const studentData = {
        name: name,
        grade: parseInt(grade),
        preferred_level: level,
        favorite_genres: selectedGenres,
        page_length: pageLength,
        publication_period: publicationPeriod,
        reading_goal: readingGoal,
        genre_diversity: genreDiversity,
        books_read: parseInt(booksRead),
        preference_genre: selectedGenres[0]
    };
    
    fetch('/register_student', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(studentData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Refresh students list
            students.push({
                student_id: data.student_id,
                name: data.student_name,
                grade: data.grade,
                favorite_genres: data.favorite_genres,
                preference_genre: data.favorite_genres[0],
                preferred_level: data.preferred_level,
                books_read: data.books_read
            });
            
            // Close modal immediately
            closeRegistrationModal();
            
            // Show success popup notification
            showSuccessPopup(`🎉 Welcome ${data.student_name}! Your ID is ${data.student_id}. You can now get personalized recommendations!`);
        } else {
            showRegistrationMessage('Registration failed. Please try again.', 'error');
            submitBtn.disabled = false;
            submitBtn.textContent = 'Register & Get Recommendations';
        }
    })
    .catch(error => {
        showRegistrationMessage('Error registering student. Please try again.', 'error');
        console.error('Error:', error);
        submitBtn.disabled = false;
        submitBtn.textContent = 'Register & Get Recommendations';
    });
}

function showRegistrationMessage(message, type) {
    const messageDiv = document.getElementById('registrationMessage');
    messageDiv.textContent = message;
    messageDiv.style.display = 'block';
    
    if (type === 'success') {
        messageDiv.style.background = '#d4edda';
        messageDiv.style.color = '#155724';
        messageDiv.style.border = '2px solid #c3e6cb';
    } else {
        messageDiv.style.background = '#f8d7da';
        messageDiv.style.color = '#721c24';
        messageDiv.style.border = '2px solid #f5c6cb';
    }
}

function showSuccessPopup(message) {
    // Create popup element
    const popup = document.createElement('div');
    popup.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px 50px;
        border-radius: 15px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.4);
        z-index: 10000;
        font-size: 1.2em;
        text-align: center;
        animation: popupFadeIn 0.3s ease-in-out;
    `;
    popup.textContent = message;
    
    // Add animation
    const style = document.createElement('style');
    style.textContent = `
        @keyframes popupFadeIn {
            from {
                opacity: 0;
                transform: translate(-50%, -50%) scale(0.8);
            }
            to {
                opacity: 1;
                transform: translate(-50%, -50%) scale(1);
            }
        }
        @keyframes popupFadeOut {
            from {
                opacity: 1;
                transform: translate(-50%, -50%) scale(1);
            }
            to {
                opacity: 0;
                transform: translate(-50%, -50%) scale(0.8);
            }
        }
    `;
    document.head.appendChild(style);
    
    // Add to page
    document.body.appendChild(popup);
    
    // Remove after 3 seconds with fade out
    setTimeout(() => {
        popup.style.animation = 'popupFadeOut 0.3s ease-in-out';
        setTimeout(() => {
            document.body.removeChild(popup);
        }, 300);
    }, 3000);
}

        // Close menu when clicking outside
        document.addEventListener('click', function(event) {
            const menu = document.getElementById('navMenu');
            const hamburger = document.querySelector('.hamburger');
            if (!menu.contains(event.target) && !hamburger.contains(event.target)) {
                menu.classList.remove('active');
            }
        });

        // Smooth scrolling for navigation
        document.querySelectorAll('.nav-menu a').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                if (this.getAttribute('href').startsWith('#')) {
                    e.preventDefault();
                    const target = document.querySelector(this.getAttribute('href'));
                    if (target) {
                        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        document.getElementById('navMenu').classList.remove('active');
                    }
                }
            });
        });

        function searchStudents() {
            const searchTerm = document.getElementById('studentSearch').value.toLowerCase();
            const resultsDiv = document.getElementById('studentSearchResults');
            
            if (searchTerm.length === 0) {
                resultsDiv.classList.remove('active');
                return;
            }

            const filtered = students.filter(s => 
                s.name.toLowerCase().includes(searchTerm) || 
                s.student_id.toLowerCase().includes(searchTerm)
            );

            if (filtered.length > 0) {
                let html = '';
                filtered.slice(0, 5).forEach(student => {
                    html += `<div class="search-item" onclick="selectStudent('${student.student_id}', '${student.name}')">${student.name} (${student.student_id})</div>`;
                });
                resultsDiv.innerHTML = html;
                resultsDiv.classList.add('active');
            } else {
                resultsDiv.innerHTML = '<div class="search-item">No students found</div>';
                resultsDiv.classList.add('active');
            }
        }

        function selectStudent(studentId, studentName) {
            selectedStudentId = studentId;
            document.getElementById('studentSearch').value = studentName;
            document.getElementById('studentSearchResults').classList.remove('active');
        }

        function searchStudentsForInsights() {
            const searchTerm = document.getElementById('insightSearch').value.toLowerCase();
            const resultsDiv = document.getElementById('insightSearchResults');
            
            if (searchTerm.length === 0) {
                resultsDiv.classList.remove('active');
                return;
            }

            const filtered = students.filter(s => 
                s.name.toLowerCase().includes(searchTerm) || 
                s.student_id.toLowerCase().includes(searchTerm)
            );

            if (filtered.length > 0) {
                let html = '';
                filtered.slice(0, 5).forEach(student => {
                    html += `<div class="search-item" onclick="selectInsightStudent('${student.student_id}', '${student.name}')">${student.name} (${student.student_id})</div>`;
                });
                resultsDiv.innerHTML = html;
                resultsDiv.classList.add('active');
            } else {
                resultsDiv.innerHTML = '<div class="search-item">No students found</div>';
                resultsDiv.classList.add('active');
            }
        }

        function selectInsightStudent(studentId, studentName) {
            selectedInsightStudentId = studentId;
            document.getElementById('insightSearch').value = studentName;
            document.getElementById('insightSearchResults').classList.remove('active');
        }

        function getRecommendations() {
            const studentId = selectedStudentId;
            const level = document.getElementById('levelFilter').value;

            if (!studentId) {
                alert('Please search and select a student');
                return;
            }

            document.getElementById('recommendationsResult').innerHTML = '<div class="loading">Loading recommendations...</div>';

            fetch(`/recommend?student_id=${studentId}&level=${level}`)
                .then(response => response.json())
                .then(data => {
                    let html = '';
                    if (data.recommendations.length === 0) {
                        html = '<div class="no-results">No recommendations available for this student with selected filters.</div>';
                    } else {
                        html += `<p style="color: #666; margin-bottom: 15px;">📖 Top 5 Hybrid Recommendations for <strong>${data.student_name}</strong>:</p>`;
                        data.recommendations.forEach((book, index) => {
                            html += `
                                <div class="book-item">
                                    <div class="book-title">${index + 1}. ${book.title}</div>
                                    <div class="book-meta">
                                        <span><strong>Author:</strong> ${book.author}</span>
                                        <span><strong>Genre:</strong> <span class="genre-badge">${book.genre}</span></span>
                                        <span><strong>Rating:</strong> <span class="rating">⭐ ${book.avg_rating}/5</span></span>
                                        <span><strong>Pages:</strong> ${book.pages}</span>
                                        <span><strong>Level:</strong> ${book.reading_level}</span>
                                        <span><strong>Year:</strong> ${book.publication_year}</span>
                                    </div>
                                </div>
                            `;
                        });
                    }
                    document.getElementById('recommendationsResult').innerHTML = html;
                })
                .catch(error => {
    console.error('Full error details:', error);
    document.getElementById('insightsResult').innerHTML = '<div class="no-results">Error loading insights: ' + error.message + '</div>';
});
        }

        function getInsights() {
    const studentId = selectedInsightStudentId;

    console.log('Selected student ID:', studentId); // DEBUG

    if (!studentId) {
        alert('Please search and select a student');
        return;
    }

    document.getElementById('insightsResult').innerHTML = '<div class="loading">Loading insights...</div>';

    fetch(`/insights?student_id=${studentId}`)
    .then(response => {
        console.log('Response status:', response.status); // DEBUG
        return response.json();
    })
    .then(data => {
        console.log('Received data:', data);
        
        if (data.error) {
            document.getElementById('insightsResult').innerHTML = '<div class="no-results">Student not found.</div>';
            return;
        }
                    if (data.error) {
                        document.getElementById('insightsResult').innerHTML = '<div class="no-results">Student not found.</div>';
                        return;
                    }

                    let html = `
                        <h3 style="color: #667eea; margin-bottom: 15px;">Reading Profile: ${data.student_name}</h3>
                        <div class="insights-grid">
                            <div class="insight-item">
                                <div class="insight-label">Grade</div>
                                <div class="insight-value">${data.grade}</div>
                            </div>
                            <div class="insight-item">
                                <div class="insight-label">Books Read</div>
                                <div class="insight-value">${data.total_books}</div>
                            </div>
                            <div class="insight-item">
                                <div class="insight-label">Total Pages</div>
                                <div class="insight-value">${data.total_pages.toLocaleString()}</div>
                                                       </div>
                            <div class="insight-item">
                                <div class="insight-label">Avg Rating Given</div>
                                <div class="insight-value">${data.avg_rating}⭐</div>
                            </div>
                            <div class="insight-item">
                                <div class="insight-label">Favorite Genre</div>
                                <div class="insight-value" style="font-size: 1.2em;">${data.favorite_genre}</div>
                            </div>
                            <div class="insight-item">
                                <div class="insight-label">Preference Genre</div>
                                <div class="insight-value" style="font-size: 1.2em;">${data.preference_genre}</div>
                            </div>
                            <div class="insight-item">
                            <div class="insight-label">Cluster Group</div>
                                <div class="insight-value" style="font-size: 1.1em;">${data.cluster_name}</div>
                            </div>
                            <div class="insight-item">
                                <div class="insight-label">Similar Students</div>
                                <div class="insight-value">${data.cluster_size}</div>
                            </div>
                        </div>
                        <h4 style="color: #667eea; margin-top: 20px; margin-bottom: 10px;">Reading Level Distribution:</h4>
                        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                            ${Object.entries(data.level_distribution).map(([level, count]) => 
                                `<span style="margin-right: 15px;"><strong>${level}:</strong> ${count} books</span>`
                            ).join('')}
                        </div>
                    `;
                    document.getElementById('insightsResult').innerHTML = html;
                })
                .catch(error => {
                    document.getElementById('insightsResult').innerHTML = '<div class="no-results">Error loading insights. Please try again.</div>';
                    console.error('Error:', error);
                });
        }

        function showAssociationRules() {
            document.getElementById('rulesModal').classList.add('active');
            
            fetch('/association_rules')
                .then(response => response.json())
                .then(data => {
                    let html = '';
                    if (data.rules.length === 0) {
                        html = '<div class="no-results">No strong association rules found in current data.</div>';
                    } else {
                        data.rules.forEach(rule => {
                            html += `
                                <div class="rule-item">
                                    <span><strong>${rule.from}</strong></span>
                                    <span class="arrow">→</span>
                                    <span><strong>${rule.to}</strong></span>
                                    <div>
                                        <span class="confidence">Confidence: ${rule.confidence}</span>
                                        <span class="confidence" style="background: #28a745; margin-left: 5px;">Support: ${rule.support}</span>
                                        <span class="confidence" style="background: #ffc107; margin-left: 5px;">Lift: ${rule.lift}</span>
                                    </div>
                                </div>
                            `;
                        });
                    }
                    document.getElementById('rulesContent').innerHTML = html;
                })
                .catch(error => {
                    document.getElementById('rulesContent').innerHTML = '<div class="no-results">Error loading rules.</div>';
                });
        }

        function closeModal() {
            document.getElementById('rulesModal').classList.remove('active');
        }

function showGenreDistribution() {
    document.getElementById('genreModal').classList.add('active');
    document.getElementById('navMenu').classList.remove('active');
}

function closeGenreModal() {
    document.getElementById('genreModal').classList.remove('active');
}

        // Close modal when clicking outside
        window.onclick = function(event) {
    const rulesModal = document.getElementById('rulesModal');
    const genreModal = document.getElementById('genreModal');
    const booksModal = document.getElementById('booksModal');
    const registrationModal = document.getElementById('registrationModal');
    const feedbackModal = document.getElementById('feedbackModal');
    const bookRatingModal = document.getElementById('bookRatingModal');
    
    if (event.target === rulesModal) rulesModal.classList.remove('active');
    if (event.target === genreModal) genreModal.classList.remove('active');
    if (event.target === booksModal) booksModal.classList.remove('active');
    if (event.target === registrationModal) registrationModal.classList.remove('active');
    if (event.target === feedbackModal) feedbackModal.classList.remove('active');
    if (event.target === bookRatingModal) bookRatingModal.classList.remove('active');
}
        // Close search results when clicking outside
        document.addEventListener('click', function(event) {
            if (!event.target.closest('.search-wrapper')) {
                document.querySelectorAll('.search-results').forEach(div => {
                    div.classList.remove('active');
                });
            }
        });

let selectedRatingStudentId = '';
let selectedRatingBookId = '';
let isEditingRating = false;

function showFeedbackForm() {
    document.getElementById('feedbackModal').classList.add('active');
    document.getElementById('navMenu').classList.remove('active');
}

function closeFeedbackModal() {
    document.getElementById('feedbackModal').classList.remove('active');
    document.getElementById('feedbackForm').reset();
    document.getElementById('feedbackRating').value = '0';
    document.querySelectorAll('.star-rating').forEach(star => star.classList.remove('active'));
    document.getElementById('feedbackMessage').style.display = 'none';
}

function setFeedbackRating(rating) {
    document.getElementById('feedbackRating').value = rating;
    document.querySelectorAll('.star-rating').forEach((star, index) => {
        if (index < rating) {
            star.classList.add('active');
            star.textContent = '★';
        } else {
            star.classList.remove('active');
            star.textContent = '☆';
        }
    });
}

function submitFeedback(event) {
    event.preventDefault();
    
    const name = document.getElementById('feedbackName').value;
    const studentId = document.getElementById('feedbackStudentId').value;
    const rating = document.getElementById('feedbackRating').value;
    const text = document.getElementById('feedbackText').value;
    
    if (!name || !text || rating === '0') {
        showFeedbackMessage('Please fill in all required fields and select a rating!', 'error');
        return;
    }
    
    const feedbackData = {
        name: name,
        student_id: studentId || 'Anonymous',
        rating: parseInt(rating),
        feedback: text,
        date: new Date().toISOString()
    };
    
    fetch('/submit_feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(feedbackData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            closeFeedbackModal();
            showSuccessPopup('🎉 Thank you for your feedback!');
        } else {
            showFeedbackMessage('Failed to submit feedback. Please try again.', 'error');
        }
    })
    .catch(error => {
        showFeedbackMessage('Error submitting feedback.', 'error');
        console.error('Error:', error);
    });
}

function showFeedbackMessage(message, type) {
    const messageDiv = document.getElementById('feedbackMessage');
    messageDiv.textContent = message;
    messageDiv.style.display = 'block';
    
    if (type === 'success') {
        messageDiv.style.background = '#d4edda';
        messageDiv.style.color = '#155724';
        messageDiv.style.border = '2px solid #c3e6cb';
    } else {
        messageDiv.style.background = '#f8d7da';
        messageDiv.style.color = '#721c24';
        messageDiv.style.border = '2px solid #f5c6cb';
    }
}

function showBookRating() {
    document.getElementById('bookRatingModal').classList.add('active');
    document.getElementById('navMenu').classList.remove('active');
}

function closeBookRatingModal() {
    document.getElementById('bookRatingModal').classList.remove('active');
    document.getElementById('bookRatingForm').reset();
    selectedRatingStudentId = '';
    selectedRatingBookId = '';
    isEditingRating = false;
    document.getElementById('bookRatingValue').value = '0';
    document.querySelectorAll('.book-star-rating').forEach(star => star.classList.remove('active'));
    document.getElementById('existingRatingNotice').style.display = 'none';
    document.getElementById('bookRatingMessage').style.display = 'none';
    document.getElementById('submitRatingBtn').textContent = 'Submit Rating';
}

function searchStudentsForRating() {
    const searchTerm = document.getElementById('ratingStudentSearch').value.toLowerCase();
    const resultsDiv = document.getElementById('ratingStudentSearchResults');
    
    if (searchTerm.length === 0) {
        resultsDiv.classList.remove('active');
        return;
    }

    const filtered = students.filter(s => 
        s.name.toLowerCase().includes(searchTerm) || 
        s.student_id.toLowerCase().includes(searchTerm)
    );

    if (filtered.length > 0) {
        let html = '';
        filtered.slice(0, 5).forEach(student => {
            html += `<div class="search-item" onclick="selectRatingStudent('${student.student_id}', '${student.name}')">${student.name} (${student.student_id})</div>`;
        });
        resultsDiv.innerHTML = html;
        resultsDiv.classList.add('active');
    } else {
        resultsDiv.innerHTML = '<div class="search-item">No students found</div>';
        resultsDiv.classList.add('active');
    }
}

function selectRatingStudent(studentId, studentName) {
    selectedRatingStudentId = studentId;
    document.getElementById('ratingStudentSearch').value = studentName;
    document.getElementById('ratingStudentSearchResults').classList.remove('active');
}

function searchBooksForRating() {
    const searchTerm = document.getElementById('ratingBookSearch').value.toLowerCase();
    const resultsDiv = document.getElementById('ratingBookSearchResults');
    
    if (searchTerm.length === 0) {
        resultsDiv.classList.remove('active');
        return;
    }

    fetch('/get_all_books')
        .then(response => response.json())
        .then(data => {
            const filtered = data.books.filter(book => 
                book.title.toLowerCase().includes(searchTerm) ||
                book.author.toLowerCase().includes(searchTerm)
            );

            if (filtered.length > 0) {
                let html = '';
                filtered.slice(0, 5).forEach(book => {
                    html += `<div class="search-item" onclick="selectRatingBook('${book.book_id}', '${book.title}')">${book.title} by ${book.author}</div>`;
                });
                resultsDiv.innerHTML = html;
                resultsDiv.classList.add('active');
            } else {
                resultsDiv.innerHTML = '<div class="search-item">No books found</div>';
                resultsDiv.classList.add('active');
            }
        });
}

function selectRatingBook(bookId, bookTitle) {
    selectedRatingBookId = bookId;
    document.getElementById('ratingBookSearch').value = bookTitle;
    document.getElementById('ratingBookSearchResults').classList.remove('active');
    
    // Check if student has already rated this book
    if (selectedRatingStudentId && selectedRatingBookId) {
        checkExistingRating();
    }
}

function checkExistingRating() {
    fetch(`/check_rating?student_id=${selectedRatingStudentId}&book_id=${selectedRatingBookId}`)
        .then(response => response.json())
        .then(data => {
            if (data.has_rating) {
                document.getElementById('existingRatingNotice').style.display = 'block';
                document.getElementById('currentRatingDisplay').textContent = '⭐'.repeat(data.rating);
                document.getElementById('submitRatingBtn').textContent = 'Update Rating';
                setBookRating(data.rating);
                document.getElementById('bookReview').value = data.review || '';
                
                // Disable editing initially
                document.querySelectorAll('.book-star-rating').forEach(star => {
                    star.style.pointerEvents = 'none';
                    star.style.opacity = '0.5';
                });
                document.getElementById('bookReview').disabled = true;
                document.getElementById('submitRatingBtn').disabled = true;
            } else {
                document.getElementById('existingRatingNotice').style.display = 'none';
                document.getElementById('submitRatingBtn').textContent = 'Submit Rating';
                document.getElementById('submitRatingBtn').disabled = false;
            }
        });
}

function enableRatingEdit() {
    isEditingRating = true;
    document.querySelectorAll('.book-star-rating').forEach(star => {
        star.style.pointerEvents = 'auto';
        star.style.opacity = '1';
    });
    document.getElementById('bookReview').disabled = false;
    document.getElementById('submitRatingBtn').disabled = false;
    document.getElementById('existingRatingNotice').querySelector('button').textContent = 'Editing...';
    document.getElementById('existingRatingNotice').querySelector('button').disabled = true;
}

function setBookRating(rating) {
    document.getElementById('bookRatingValue').value = rating;
    document.querySelectorAll('.book-star-rating').forEach((star, index) => {
        if (index < rating) {
            star.classList.add('active');
            star.textContent = '★';
        } else {
            star.classList.remove('active');
            star.textContent = '☆';
        }
    });
}

function submitBookRating(event) {
    event.preventDefault();
    
    if (!selectedRatingStudentId || !selectedRatingBookId) {
        showBookRatingMessage('Please select both student and book!', 'error');
        return;
    }
    
    const rating = document.getElementById('bookRatingValue').value;
    if (rating === '0') {
        showBookRatingMessage('Please select a rating!', 'error');
        return;
    }
    
    const ratingData = {
        student_id: selectedRatingStudentId,
        book_id: selectedRatingBookId,
        rating: parseInt(rating),
        review: document.getElementById('bookReview').value || '',
        date_read: new Date().toISOString().split('T')[0]
    };
    
    const submitBtn = document.getElementById('submitRatingBtn');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Submitting...';
    
    fetch('/submit_book_rating', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(ratingData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            closeBookRatingModal();
            showSuccessPopup(data.message);
        } else {
            showBookRatingMessage(data.message || 'Failed to submit rating.', 'error');
            submitBtn.disabled = false;
            submitBtn.textContent = 'Submit Rating';
        }
    })
    .catch(error => {
        showBookRatingMessage('Error submitting rating.', 'error');
        console.error('Error:', error);
        submitBtn.disabled = false;
        submitBtn.textContent = 'Submit Rating';
    });
}

function showBookRatingMessage(message, type) {
    const messageDiv = document.getElementById('bookRatingMessage');
    messageDiv.textContent = message;
    messageDiv.style.display = 'block';
    
    if (type === 'success') {
        messageDiv.style.background = '#d4edda';
        messageDiv.style.color = '#155724';
        messageDiv.style.border = '2px solid #c3e6cb';
    } else {
        messageDiv.style.background = '#f8d7da';
        messageDiv.style.color = '#721c24';
        messageDiv.style.border = '2px solid #f5c6cb';
    }
}

    </script>
</body>
</html>
'''
@app.route('/')
def index():
    avg_rating = reading_history_df['rating'].mean()
    
    cluster_names = {
        0: "Casual Readers",
        1: "Diverse Explorers",
        2: "Genre Enthusiasts",
        3: "Advanced Scholars",
        4: "Avid Bookworms"
    }

    cluster_stats = []
    for i in range(n_clusters):
        cluster_count = (student_features_df['cluster'] == i).sum()
        cluster_stats.append({
            'id': i, 
            'name': cluster_names[i],
            'count': cluster_count
        })
    
    genre_counts = books_df['genre'].value_counts().to_dict()
    genre_distribution = []
    for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = int((count / len(books_df)) * 100)
        genre_distribution.append({'name': genre, 'count': count, 'percentage': percentage})
    
    stats = {
        'total_books': len(books_df),
        'total_students': len(students_df),
        'avg_rating': round(avg_rating, 1),
        'total_records': len(reading_history_df)
    }
    
    return render_template_string(HTML_TEMPLATE, 
                                 students=students_df.to_dict('records'),
                                 stats=stats,
                                 cluster_stats=cluster_stats,
                                 genre_distribution=genre_distribution)


@app.route('/recommend')
def recommend():
    student_id = request.args.get('student_id')
    level = request.args.get('level') or None
    
    recommendations = get_hybrid_recommendations(student_id, num_recommendations=5, reading_level_filter=level)
    student_name = students_df[students_df['student_id'] == student_id]['name'].iloc[0] if student_id in students_df['student_id'].values else 'Unknown'
    
    return jsonify({
        'student_name': student_name,
        'recommendations': recommendations
    })


@app.route('/insights')
def insights():
    student_id = request.args.get('student_id')
    print(f"Received student_id: {student_id}")  # DEBUG
    print(f"Available student IDs: {students_df['student_id'].tolist()[:5]}")  # DEBUG
    
    insights_data = get_student_insights(student_id)
    
    if insights_data is None:
        print(f"Student {student_id} not found")  # DEBUG
        return jsonify({'error': 'Student not found'})
    
    print(f"Returning insights: {insights_data}")  # DEBUG
    return jsonify(insights_data)


@app.route('/association_rules')
def association_rules_route():
    rules = calculate_association_rules()
    return jsonify({'rules': rules})

@app.route('/get_all_books')
def get_all_books():
    books_list = books_df.to_dict('records')
    return jsonify({'books': books_list})

@app.route('/register_student', methods=['POST'])
def register_student():
    global students_df, student_features_df
    
    data = request.json
    
    # Generate new student ID
    existing_ids = students_df['student_id'].tolist()
    last_id_num = max([int(sid[1:]) for sid in existing_ids])
    new_student_id = f"S{last_id_num + 1:02d}"
    
    # Create new student record with all new fields
    new_student = {
        'student_id': new_student_id,
        'name': data['name'],
        'grade': data['grade'],
        'preference_genre': data.get('preference_genre', data['favorite_genres'][0] if data.get('favorite_genres') else 'Fiction'),
        'favorite_genres': data.get('favorite_genres', []),
        'preferred_level': data['preferred_level'],
        'books_read': data['books_read'],
        'page_length': data.get('page_length'),
        'publication_period': data.get('publication_period'),
        'reading_goal': data.get('reading_goal'),
        'genre_diversity': data.get('genre_diversity')
    }
    
    # Add to DataFrame (in memory)
    students_df = pd.concat([students_df, pd.DataFrame([new_student])], ignore_index=True)
    
    # Create initial feature entry for the new student
    new_features = {
        'student_id': new_student_id,
        'avg_rating_given': 0,
        'genre_diversity': 0,
        'avg_pages_read': 0,
        'favorite_genre_count': 0,
        'beginner_count': 0,
        'intermediate_count': 0,
        'advanced_count': 0,
        'books_read': 0,
        'avg_page_preference': 300,
        'modern_vs_classic_ratio': 1,
        'genre_diversity_score': 0,
        'cluster': np.random.randint(0, n_clusters)
    }
    
    student_features_df = pd.concat([student_features_df, pd.DataFrame([new_features])], ignore_index=True)
    
    return jsonify({
        'success': True,
        'student_id': new_student_id,
        'student_name': data['name'],
        'grade': data['grade'],
        'preference_genre': new_student['preference_genre'],
        'favorite_genres': data.get('favorite_genres', []),
        'preferred_level': data['preferred_level'],
        'books_read': data['books_read']
    })

# At the top of your file, after other imports
feedbacks_list = []
book_ratings_list = []

# Add these routes before if __name__ == '__main__':

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    global feedbacks_list
    
    data = request.json
    feedback = {
        'id': len(feedbacks_list) + 1,
        'name': data['name'],
        'student_id': data['student_id'],
        'rating': data['rating'],
        'feedback': data['feedback'],
        'date': data['date']
    }
    
    feedbacks_list.append(feedback)
    
    return jsonify({'success': True, 'message': 'Feedback submitted successfully!'})


@app.route('/check_rating')
def check_rating():
    global reading_history_df
    
    student_id = request.args.get('student_id')
    book_id = request.args.get('book_id')
    
    # Check if rating exists in reading_history_df
    existing_rating = reading_history_df[
        (reading_history_df['student_id'] == student_id) & 
        (reading_history_df['book_id'] == book_id)
    ]
    
    if len(existing_rating) > 0:
        rating_value = int(existing_rating.iloc[0]['rating'])
        return jsonify({
            'has_rating': True,
            'rating': rating_value,
            'review': ''  # We don't have reviews in original data
        })
    else:
        return jsonify({'has_rating': False})


@app.route('/submit_book_rating', methods=['POST'])
def submit_book_rating():
    global reading_history_df, book_ratings_list
    
    data = request.json
    student_id = data['student_id']
    book_id = data['book_id']
    
    # Check if rating already exists
    existing_mask = (reading_history_df['student_id'] == student_id) & (reading_history_df['book_id'] == book_id)
    
    if existing_mask.any():
        # Update existing rating
        reading_history_df.loc[existing_mask, 'rating'] = data['rating']
        reading_history_df.loc[existing_mask, 'date_read'] = data['date_read']
        message = '✅ Your rating has been updated successfully!'
    else:
        # Add new rating
        new_rating = pd.DataFrame([{
            'student_id': student_id,
            'book_id': book_id,
            'rating': data['rating'],
            'date_read': data['date_read']
        }])
        reading_history_df = pd.concat([reading_history_df, new_rating], ignore_index=True)
        message = '✅ Thank you for rating this book!'
    
    # Store detailed review separately
    book_ratings_list.append({
        'student_id': student_id,
        'book_id': book_id,
        'rating': data['rating'],
        'review': data['review'],
        'date': data['date_read']
    })
    
    return jsonify({'success': True, 'message': message})

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
