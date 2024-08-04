# Import necessary modules
from flask import Flask, render_template, request
from markupsafe import escape
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask_caching import Cache
from werkzeug.utils import secure_filename
from loc import read_pdf
import os

# Disable the symlink warning to prevent unnecessary warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Initialize Flask app and Flask-Caching
app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Define a function for property terms summarization
def property_terms_summarization(text, target_lines=7, importance_threshold=0.1):
    try:
        # Load pre-trained model and tokenizer from Hugging Face Transformers
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/bart-large-cnn")

        # Split the text into chunks of 512 tokens (model's maximum input length)
        chunks = [text[i:i+512] for i in range(0, len(text), 512)]

        # Initialize an empty summary
        summary = ""

        # Iterate over each chunk
        for chunk in chunks:
            # Tokenize and generate summary for each chunk
            inputs = tokenizer(
                "summarize: " + chunk, return_tensors="pt", max_length=2048, truncation=True)
            summary_ids = model.generate(
                inputs['input_ids'],
                max_length=2048,
                min_length=1,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )

            # Decode the summary and append to the overall summary
            decoded_summary = tokenizer.decode(
                summary_ids[0], skip_special_tokens=True)
            summary += decoded_summary

        # Limit the summary to the target number of lines
        summary_lines = summary.split('\n')[:target_lines]
        final_summary = '\n'.join(summary_lines)

        return final_summary
    
    except Exception as e:
        # Log any errors that occur during summarization
        app.logger.error(f"Error during property terms summarization: {e}")
        return f"Error during property terms summarization: {e}. Please try again."

# Define the route for the main page
@app.route('/')
def index():
    # Render the main page template
    return render_template('index.html')

# Define the route for summarization (handles POST requests)
@app.route('/', methods=['POST'])
def summarize():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' in request.files:
            uploaded_file = request.files['file']
            if uploaded_file.filename != '':
                # Read text from the uploaded PDF file
                pdf_text = read_pdf(uploaded_file)
                if pdf_text:
                    # Perform summarization using the extracted PDF text
                    summary = property_terms_summarization(pdf_text)
                    # Render the main page template with the summary
                    return render_template('index.html', summary=summary)

        # If no file was uploaded or extraction failed, render the main page template
        return render_template('index.html')


# Run the Flask application if the script is executed directly
if __name__ == '__main__':
    app.run(debug=True)
