from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
ingredients = "milk, oats, banana, honey"

labels = ["Protein", "Carbohydrates", "Fats", "Vitamins", "Iron"]
result = classifier(ingredients, labels)
print("Detected Nutrients:", result["labels"])