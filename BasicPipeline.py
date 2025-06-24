from transformers import pipeline

qa_pipeline = pipeline("question-answering")

context = """Charles Babbage was an English mathematician and inventor. 
He is credited with conceptualizing and inventing the first mechanical computer."""

question = "Who invented the first mechanical computer?"

result = qa_pipeline(question=question, context=context)
print(result)
