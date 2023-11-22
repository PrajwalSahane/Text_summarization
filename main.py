from flask import Flask, render_template, request
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Disable the symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

app = Flask(__name__)

def Descriptive_summarization(text, min_percentage=10, max_percentage=50):
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

    # Set dynamic min_summary_length and max_summary_length based on input length
    input_length = len(tokenizer.encode(text, return_tensors='pt')[0])
    dynamic_min_length = max(1, int(input_length * min_percentage / 100))
    dynamic_max_length = int(input_length * max_percentage / 100)

    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=2048, truncation=True)

    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=min(dynamic_max_length, 2048),  # Limit to 2048 for BART model
        min_length=dynamic_min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@app.route('/')
def index():
    print("Index function called")
    return render_template('index.html')

@app.route('/', methods=['POST'])
def summarize():
    if request.method == 'POST':
        user_input = request.form['text_input']
        print("User input:", user_input)
        user_input_html = user_input.replace('\n', '<br>')
        Descriptive_summary = Descriptive_summarization(user_input)
        print("Descriptive Summary:", Descriptive_summary)
        return render_template('index.html', original_text=user_input_html, summary=Descriptive_summary)

if __name__ == '__main__':
    app.run(debug=True)
