from flask import Flask, render_template, request
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import re

app = Flask(__name__)

# Load the pre-trained model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
model.load_state_dict(torch.load('bart_cocktail_model.pt', map_location=torch.device('cpu')))

@app.route("/return", methods=["POST"])
def move_BBBB():
    return render_template("index.html")

@app.route('/')
def home():
    return render_template('index.html')


def parse_generated_text_v5(generated_text):
    # Split the generated text based on spaces
    # elements = generated_text[0].split('[')
    if isinstance(generated_text, list):
        generated_text = generated_text[0]

    elements = generated_text.split('[')
    
    # Extract each component
    technique = elements[0].strip()
    
    ingredients = [item.strip(" '") for item in elements[1].rstrip(" ']").split(',')]
    amounts = [item.strip(" '") for item in elements[2].rstrip(" ']").split(',')]
    units = [item.strip(" '") for item in elements[3].rstrip(" ']").split(',')]
    
    # Ensure that the lengths of ingredients, amounts, and units are the same
    while len(ingredients) > len(amounts):
        amounts.append('')
    while len(ingredients) > len(units):
        units.append('')
    
    return {
        "技法": technique,
        "材料": ingredients,
        "分量": amounts,
        "単位": units
    }



def combine_ingredients_amounts_units_final(ingredients, amounts, units):
    combined = []
    for i, (ingredient, amount, unit) in enumerate(zip(ingredients, amounts, units)):
        if not unit:  # Check if unit is empty
            amount = "0"
            unit = "適量"
        combined.append((ingredient, amount, unit))
    return combined



@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form and combine into a single string
    taste = request.form.get('taste')
    alcohol_content = request.form.get('alcohol_content')
    alcohol_type = request.form.get('alcohol_type')
    base = request.form.get('base')
    user_input = ', '.join([taste, alcohol_content, alcohol_type, base])

    # Encode the input data
    new_input_encodings = tokenizer([user_input], truncation=True, padding=True, return_tensors='pt')

    # Model prediction
    with torch.no_grad():
        outputs = model.generate(new_input_encodings.input_ids, max_length=300)

    # Decode the generated text
    generated_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in outputs]

 # Convert the generated text to the desired dictionary format
    # parsed_data = parse_generated_text_v4(generated_text)
    parsed_data = parse_generated_text_v5(generated_text)

    # Combine ingredients, amounts, and units using the final function
    combined = combine_ingredients_amounts_units_final(
        parsed_data["材料"],
        parsed_data["分量"],
        parsed_data["単位"]
    )

    # print("Combined data:", combined)


    # Return the parsed data as a response
    return render_template('result.html', prediction=parsed_data, combined=combined)




if __name__ == '__main__':
    app.run(debug=False)
