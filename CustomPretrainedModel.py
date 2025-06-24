from transformers import pipeline

# Define the context and question BEFORE using them
context = """Charles Babbage was an English mathematician and inventor. 
He is credited with conceptualizing and inventing the first mechanical computer."""

question = "Who invented the first mechanical computer?"

# Load custom QA model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Run QA pipeline
result = qa_pipeline(question=question, context=context)

# Print result
print(result)
