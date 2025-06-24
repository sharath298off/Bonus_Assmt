from transformers import pipeline

# Step 1: Initialize the question answering pipeline
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Step 2: Define your own context
my_context = """The Eiffel Tower was constructed in 1889 in Paris, France.
It remains one of the most iconic landmarks in the world."""

# Step 3: Define two different questions
questions = [
    "When was the Eiffel Tower constructed?",
    "Where is the Eiffel Tower located?"
]

# Step 4: Run the QA model on each question
for q in questions:
    result = qa_pipeline(question=q, context=my_context)
    print(f"Question: {q}")
    print(f"Answer: {result['answer']}, Score: {result['score']:.2f}\n")
