# Final reflection

The six labs went from counting words to building an agent that searches documents and adds events to my calendar. Looking back, each lab answered a problem the one before it had left open, and most of what I learned came from the moments when something didn't work the way I expected.

## From word counts to attention

In Lab 1 my Bag of Words models reached about 82% accuracy on 3,000 review sentences, and Logistic Regression and the SVM were only 2 test reviews apart. The numbers looked fine until I typed in my own sentences. "This movie was not bad" came out negative, because Bag of Words only counts words and ignores their order, so "not bad" is just "not" plus "bad".

Lab 3 was the answer to that. Writing the Transformer myself, one class at a time, showed me where word order comes back in. Position embeddings give the model the order of the characters, and masked self-attention lets each position look at everything before it. My tiny model only had 113,096 parameters and a 64-character window, so it learned what the stories look like (capital letters, names, "Once upon a time") but not grammar or meaning. After seeing that, the size of real LLMs made more sense to me. The layer and head experiments also showed the trade-off in practice: 4 layers gave a lower loss than 1, but took more than three times as long to train.

## Three ways to adapt a model

Labs 2, 4 and 5 used three different ways of getting a model to do what I need, and comparing them is the most useful thing I'm taking from the module.

- **Prompting (Lab 2).** With Copilot as a product-development partner, a clear prompt with a role, some context and a limit on the answer was enough. The answers got more specific as I gave more detail, and the model kept the earlier steps in mind because everything was in one chat. It gave me lots of options quickly, but I still made the decisions. Its favourite idea was the tags on their own; I combined them with a timetable app instead.
- **Fine-tuning (Lab 4).** QLoRA trained only 0.22% of Qwen2.5-0.5B's parameters, in 42 seconds on a free Colab GPU, and saved an adapter of about 4 MB. It changed how the model answers far more than what it knows. The answers dropped from about 100 words to one sentence, like my dataset, but some details from my answers didn't come through. With 30 examples, fine-tuning teaches style more than facts.
- **RAG (Lab 5).** Retrieval gave a 3B model facts from the NSBM guide that it could never have known, and the "use only the context" prompt stopped it from guessing. But the answer could only be as good as the chunks the search returned. A module table cut across two chunks gave answers with 11 of the 15 modules, and asking about "subjects" instead of "modules" made it refuse. In the group activity, retrieving whole pages kept the table together, which showed me that the best chunking depends on how the document is laid out.

So for a new problem, I would start with prompting, add RAG when the model needs facts it doesn't have, and use fine-tuning when I need the answers in a particular style or format.

## Agents

Lab 6 put these pieces together. A small local model chose the right tool for all six test queries, so tool choice turned out to be the easy part. Almost all my fixes were about what the model saw after the tool ran. It reported a file name, `123.pdf`, as the number of files it found, its memory step saved its own replies as facts about me, and a bare "21" from the calculator was too little for it to use. Labelling each tool result and building one clear prompt for the final answer fixed most of these. I also reused my Lab 5 RAG as a tool, and the same retrieval problems came with it.

## Checking the output

Many problems that looked like model problems at first were really data problems:

- In Lab 1 a quote character made pandas load 748 IMDb reviews instead of 1,000.
- In Lab 3 2,237 broken characters had taken 8 slots in the vocabulary.
- In Lab 5 the PDF's fonts turned "Software" into "So�ware".

I learned to check the data before trusting a result: compare row counts with the file, print the chunks the search returned, and check the loss before training against what random guessing would give.

I also learned to make comparisons fair. In Lab 4 both models got the same questions with greedy decoding, and the original model's answers were saved before the LoRA adapter was added. In Lab 3 every experiment used the same seed and text. In the Lab 5 group activity only the context changed between the three chunking methods.

Generated output can also look finished and still be wrong. The product photo in Lab 2 says "All Set!" next to an unticked checklist, and the Lab 6 agent once listed a calendar event that wasn't in the tool result. Checking what an LLM produces is part of using it.

## Tools and skills

By the end of the module I had used pandas, scikit-learn, PyTorch, Hugging Face `datasets` and `transformers`, PEFT and TRL, `bitsandbytes`, Ollama, ChromaDB, Streamlit, LangChain, SQLite and the Google Calendar API. Some of the hardest problems had nothing to do with the models:

- Getting Lab 4 to train on a T4 meant learning that the GPU has no bfloat16, and loading the model in float32 instead.
- Lab 6 needed a full OAuth setup, because an API key can't reach a private calendar.
- VS Code twice saved an open file over my changes, and Windows wouldn't delete a database a script still had open.

Running models locally on a laptop CPU was slow (a median of 29 seconds for each RAG answer), but my questions and the university PDFs never left my computer.

## Using LLMs responsibly

The repository is public, so I kept `credentials.json`, `token.json` and the memory database out of Git, and moved one test away from my Downloads folder because the screenshot showed private file names. I noted the licences of the UCI and TinyStories datasets. In Lab 2 I wrote down which parts came from Copilot, which from Claude and which were my own idea. The agent's tools check their inputs before running anything, because the model's arguments can be wrong.

## What I would do next

- Train the Lab 3 model for longer on a GPU, with a subword tokenizer.
- Test the Lab 4 adapter on more new questions, and judge the answers by meaning, not only by character similarity.
- Use parent-child retrieval in the main RAG app, and rewrite the search query before retrieving, since the query's wording caused most of the RAG and agent failures.

The module changed how I see LLMs. I started out thinking of them as one model that answers questions. Now I see a system: the data, the tokenizer, the retrieval and the prompts around the model all decide whether the answer is right.
