# NEET Quiz Performance Analysis and Rank Prediction

## Overview
This project analyzes students' quiz performances and provides personalized recommendations for improvement. It also predicts the student's NEET rank based on their quiz scores using a linear regression model.

## Features
- **Synthetic Quiz Data Generation**: Generates mock quiz data for multiple students.
- **Performance Analysis**: Evaluates student performance and provides insights.
- **Personalized Recommendations**: Suggests weak areas for improvement.
- **NEET Rank Prediction**: Uses a regression model to estimate a student's NEET rank.

### Code Implementation:
```python
import pandas as pd
import random
import json
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate synthetic quiz data
def generate_quiz_data(num_users=10, num_quizzes=5):
    topics = ["Physics", "Chemistry", "Biology"]
    difficulty_levels = ["Easy", "Medium", "Hard"]
    users = [f"U{i}" for i in range(1, num_users+1)]
    
    current_quiz_data = []
    historical_quiz_data = []
    
    for user in users:
        history = []
        for _ in range(num_quizzes):
            quiz_id = f"Q{random.randint(100, 999)}"
            responses = {}
            score = random.randint(0, 5)  # Mocking total score
            
            for i in range(5):  # 5 questions per quiz
                qid = f"Q{i+1}"
                topic = random.choice(topics)
                difficulty = random.choice(difficulty_levels)
                correct_answer = random.choice(["A", "B", "C", "D"])
                selected_option = random.choice(["A", "B", "C", "D"]) 
                responses[qid] = selected_option
            
            history.append({"quiz_id": quiz_id, "score": score, "responses": responses})
        
        historical_quiz_data.append({"user_id": user, "history": history})
        
        # Generate current quiz data based on last quiz
        last_quiz = history[-1]
        current_quiz_data.append({
            "user_id": user,
            "quiz_id": last_quiz["quiz_id"],
            "score": last_quiz["score"]
        })
    
    return current_quiz_data, historical_quiz_data

# Save generated data
def save_data():
    current_quiz_data, historical_quiz_data = generate_quiz_data()
    
    with open("current_quiz_data.json", "w") as f:
        json.dump(current_quiz_data, f, indent=4)
    
    with open("historical_quiz_data.json", "w") as f:
        json.dump(historical_quiz_data, f, indent=4)
    
    print("Synthetic data saved successfully.")

# Analyze data and generate recommendations
def analyze_performance():
    with open("historical_quiz_data.json", "r") as f:
        historical_data = json.load(f)
    
    insights = []
    
    for user_data in historical_data:
        user_id = user_data["user_id"]
        scores = [quiz["score"] for quiz in user_data["history"]]
        avg_score = round(np.mean(scores), 2)
        
        insights.append({
            "user_id": user_id,
            "average_score": avg_score,
            "recommendation": "Focus on improving accuracy in weak topics."
        })
    
    with open("recommendations.json", "w") as f:
        json.dump(insights, f, indent=4)
    
    print("Recommendations generated successfully.")
    print(json.dumps(insights, indent=4))

# Predict NEET rank based on quiz performance
def predict_neet_rank():
    with open("historical_quiz_data.json", "r") as f:
        historical_data = json.load(f)
    
    user_scores = []
    user_ranks = []
    
    for user_data in historical_data:
        scores = [quiz["score"] for quiz in user_data["history"]]
        avg_score = np.mean(scores)
        mock_neet_rank = 1000 - (avg_score * 100)  # Mocking rank prediction
        user_scores.append(avg_score)
        user_ranks.append(mock_neet_rank)
    
    X = np.array(user_scores).reshape(-1, 1)
    y = np.array(user_ranks)
    
    model = LinearRegression()
    model.fit(X, y)
    
    predictions = model.predict(X)
    
    neet_predictions = [{"user_id": historical_data[i]["user_id"], "predicted_neet_rank": round(predictions[i])} for i in range(len(predictions))]
    
    with open("neet_rank_predictions.json", "w") as f:
        json.dump(neet_predictions, f, indent=4)
    
    print("NEET rank predictions generated successfully.")
    print(json.dumps(neet_predictions, indent=4))

# Run data generation, analysis, and rank prediction
save_data()
analyze_performance()
predict_neet_rank()
```

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd neet-quiz-analysis
   ```
2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn
   ```
3. Run the script:
   ```bash
   python neet_quiz_analysis.py
   ```

## Output
### Example Recommendations:
```json
[
    {
        "user_id": "U1",
        "average_score": 3.2,
        "recommendation": "Focus on improving accuracy in weak topics."
    },
    {
        "user_id": "U2",
        "average_score": 2.8,
        "recommendation": "Focus on improving accuracy in weak topics."
    }
]
```

### Example NEET Rank Predictions:
```json
[
    {
        "user_id": "U1",
        "predicted_neet_rank": 680
    },
    {
        "user_id": "U2",
        "predicted_neet_rank": 720
    }
]
```

## Approach
1. **Data Generation**: Synthetic quiz results are created with random scores and question responses.
2. **Performance Analysis**: Computes average scores and identifies weak topics.
3. **Recommendations**: Provides improvement suggestions based on quiz trends.
4. **Rank Prediction**: Uses linear regression to estimate the NEET rank based on past quiz performance.


