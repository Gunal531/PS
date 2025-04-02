from flask import Flask, render_template, request
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from scipy.stats import binom, poisson, norm, uniform, geom, expon
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
nlp = spacy.load("en_core_web_sm")

# Load Dataset from CSV
def create_dataset_from_csv():
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.csv')
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return pd.DataFrame(columns=["question", "distribution"])

# Train model
def train_model(custom_df=None):
    df = custom_df if custom_df is not None else create_dataset_from_csv()
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(df["question"], df["distribution"])
    return model

model = train_model()

def classify_distribution(question):
    return model.predict([question])[0]

def extract_numbers(question):
    doc = nlp(question)
    return [float(token.text) for token in doc if token.like_num]

def solve_distribution_step_by_step(question, distribution):
    steps = []
    numbers = extract_numbers(question)
    steps.append(f"🔍 Step 1: Identified Distribution → {distribution} Distribution")

    try:
        if distribution == "Binomial" and len(numbers) >= 3:
            n, p_value, k = int(numbers[0]), numbers[1], int(numbers[2])
            p = p_value / 100 if p_value > 1 else p_value
            q = 1 - p
            comb = binom.pmf(k, n, p) / (p**k * q**(n-k))
            steps.append(f"✅ Why Binomial? Fixed trials (n={n}), constant p={p}, independent trials.")
            steps.append(f"📐 Step 2: Formula → P(X=k) = C(n,k) * p^k * (1-p)^(n-k)")
            steps.append(f"🔢 Step 3: Substitution → C({n},{k}) * ({p})^{k} * ({q})^{n-k}")
            steps.append(f"🔢 Step 4: Calculation → C({n},{k}) ≈ {int(comb)}, Probability ≈ {binom.pmf(k,n,p):.4f}")
            return "\n".join(steps) + f"\n✅ Final Answer: {binom.pmf(k,n,p):.4f}"

        if distribution == "Poisson" and len(numbers) >= 2:
            lambda_, k = numbers[0], int(numbers[1])
            steps.append(f"✅ Why Poisson? Fixed time interval, mean rate λ={lambda_}, count of events k={k}")
            steps.append("📐 Step 2: Formula → P(X=k) = (λ^k * e^-λ) / k!")
            prob = poisson.pmf(k, mu=lambda_)
            steps.append(f"🔢 Step 3: Substitution → P(X={k}) = ({lambda_}^{k} * e^-{lambda_}) / {k}!")
            steps.append(f"🔢 Step 4: Calculation → Probability ≈ {prob:.4f}")
            return "\n".join(steps) + f"\n✅ Final Answer: {prob:.4f}"

        if distribution == "Normal" and len(numbers) >= 3:
            mu, sigma, x = numbers[0], numbers[1], numbers[2]
            prob = 1 - norm.cdf(x, loc=mu, scale=sigma)
            steps.append(f"✅ Why Normal? Continuous data, symmetric around mean μ={mu}, σ={sigma}")
            steps.append("📐 Step 2: Formula → P(X > x) = 1 - Φ(x)")
            steps.append(f"🔢 Step 3: Substitution → P(X > {x}) = 1 - norm.cdf({x}, loc={mu}, scale={sigma})")
            steps.append(f"🔢 Step 4: Calculation → Probability ≈ {prob:.4f}")
            return "\n".join(steps) + f"\n✅ Final Answer: {prob:.4f}"

        if distribution == "Uniform" and len(numbers) >= 3:
            a, b, x = numbers[0], numbers[1], numbers[2]
            prob = (b - x) / (b - a)
            steps.append(f"✅ Why Uniform? Equal probability between range a={a}, b={b}")
            steps.append("📐 Step 2: Formula → P(X > x) = (b - x) / (b - a)")
            steps.append(f"🔢 Step 3: Substitution → ({b} - {x}) / ({b} - {a})")
            steps.append(f"🔢 Step 4: Calculation → Probability ≈ {prob:.4f}")
            return "\n".join(steps) + f"\n✅ Final Answer: {prob:.4f}"

        if distribution == "Geometric" and len(numbers) >= 2:
            p_value, k = numbers[0], int(numbers[1])
            p = p_value / 100 if p_value > 1 else p_value
            steps.append(f"✅ Why Geometric? First success in k trials, success probability p={p}")
            steps.append("📐 Step 2: Formula → P(X=k) = (1-p)^(k-1) * p")
            prob = geom.pmf(k, p)
            steps.append(f"🔢 Step 3: Substitution → (1-{p})^{k-1} * {p}")
            steps.append(f"🔢 Step 4: Calculation → Probability ≈ {prob:.4f}")
            return "\n".join(steps) + f"\n✅ Final Answer: {prob:.4f}"

        if distribution == "Exponential" and len(numbers) >= 2:
            mean, x = numbers[0], numbers[1]
            rate = 1 / mean
            prob = 1 - expon.cdf(x, scale=mean)
            steps.append(f"✅ Why Exponential? Time between events, continuous variable, rate=1/mean={rate:.4f}")
            steps.append("📐 Step 2: Formula → P(X > x) = e^(-λx)")
            steps.append(f"🔢 Step 3: Substitution → e^(-{rate:.4f} * {x})")
            steps.append(f"🔢 Step 4: Calculation → Probability ≈ {prob:.4f}")
            return "\n".join(steps) + f"\n✅ Final Answer: {prob:.4f}"

    except Exception as e:
        steps.append(f"❌ Error in solving: {str(e)}")

    return "\n".join(steps) + "\n❌ Final Answer: Solution not found"

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    distribution = None
    question = ""
    if request.method == 'POST':
        question = request.form['question']
        distribution = classify_distribution(question)
        result = solve_distribution_step_by_step(question, distribution)
    return render_template('index.html', distribution=distribution, result=result, question=question)

if __name__ == '__main__':
    model = train_model(custom_df=create_dataset_from_csv())
    app.run(debug=True)
    
##
