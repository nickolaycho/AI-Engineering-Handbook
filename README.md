---
title: AI Engineering Handbook
---

- [Contents](#contents)
- [What is AI Engineering?](#what-is-ai-engineering)
  - [AI machine analogy](#ai-machine-analogy)
- [Large Language Models](#large-language-models)
  - [What is an LLM?](#what-is-an-llm)
  - [LLM Architecture](#llm-architecture)
    - [Tokens](#tokens)
    - [Word embeddings](#word-embeddings)
    - [Attention & Transformer](#attention-transformer)
  - [LLM Development](#llm-development)
  - [LLM Evaluation](#llm-evaluation)
    - [Metrics to evaluate an LLM](#metrics-to-evaluate-an-llm)
    - [Other ways to evaluate LLMs](#other-ways-to-evaluate-llms)
  - [Current LLMs limitations](#current-llms-limitations)
- [How to build applications on top of a Foundational
  LLM](#how-to-build-applications-on-top-of-a-foundational-llm)
  - [LLM Application base
    architecture](#llm-application-base-architecture)
    - [3<sup>rd</sup> party cloud-hosted LLM
      ](#rd-party-cloud-hosted-llm)
    - [Open-source/proprietary LLM](#open-sourceproprietary-llm)
    - [3<sup>rd</sup> party model vs
      open-source/proprietary](#rd-party-model-vs-open-sourceproprietary)
  - [How to choose the right Foundation
    Model](#how-to-choose-the-right-foundation-model)
- [Prompting strategies](#prompting-strategies)
- [Reasoning in LLMs](#reasoning-in-llms)
  - [What is reasoning for LLMs?](#what-is-reasoning-for-llms)
  - [LLM reasoning techniques](#llm-reasoning-techniques)
    - [Training-time techniques](#training-time-techniques)
    - [Inference-time techniques](#inference-time-techniques)
- [Retrieval-Augmented Generation
  (RAG)](#retrieval-augmented-generation-rag)
- [Model Context Protocol (MCP)](#model-context-protocol-mcp)
  - [MCP Architecture](#mcp-architecture)
  - [MCP vs API](#mcp-vs-api)
- [AI Agents](#ai-agents)
  - [What is an Agent?](#what-is-an-agent)
  - [Some issues with AI Agents:](#some-issues-with-ai-agents)
  - [Context Engineering for AI
    Agents](#context-engineering-for-ai-agents)
  - [Key-Value (KV) Cache
    Optimization](#key-value-kv-cache-optimization)
- [Finetuning](#finetuning)
  - [Model distillation](#model-distillation)
  - [Finetuning vs RAG](#finetuning-vs-rag)
  - [Parameter-Efficient Finetuning
    (PEFT)](#parameter-efficient-finetuning-peft)
  - [Multi-task finetuning](#multi-task-finetuning)
  - [Finetuning workflow](#finetuning-workflow)
    - [Development path](#development-path)
    - [Distillation path](#distillation-path)
- [Adapting a model to a task](#adapting-a-model-to-a-task)
- [Other](#other)
  - [Quantization](#quantization)
  - [ Small Language Models](#small-language-models)
  - [Security](#security)
- [Bonus](#bonus)

By Nicola Orecchini

# Contents

[1 What is AI Engineering?
[4](#what-is-ai-engineering)](#what-is-ai-engineering)

[1.1 AI machine analogy [5](#ai-machine-analogy)](#ai-machine-analogy)

[2 Large Language Models
[6](#large-language-models)](#large-language-models)

[2.1 What is an LLM? [6](#what-is-an-llm)](#what-is-an-llm)

[2.2 LLM Architecture [6](#llm-architecture)](#llm-architecture)

[2.2.1 Tokens [6](#tokens)](#tokens)

[2.2.2 Word embeddings [7](#word-embeddings)](#word-embeddings)

[2.2.3 Attention & Transformer
[9](#attention-transformer)](#attention-transformer)

[2.3 LLM Development [11](#llm-development)](#llm-development)

[2.4 LLM Evaluation [11](#llm-evaluation)](#llm-evaluation)

[2.4.1 Metrics to evaluate an LLM
[12](#metrics-to-evaluate-an-llm)](#metrics-to-evaluate-an-llm)

[2.4.2 Other ways to evaluate LLMs
[13](#other-ways-to-evaluate-llms)](#other-ways-to-evaluate-llms)

[2.5 Current LLMs limitations [13](#_Toc218633902)](#_Toc218633902)

[3 How to build applications on top of a Foundational LLM
[15](#how-to-build-applications-on-top-of-a-foundational-llm)](#how-to-build-applications-on-top-of-a-foundational-llm)

[3.1 LLM Application base architecture
[15](#llm-application-base-architecture)](#llm-application-base-architecture)

[3.1.1 3<sup>rd</sup> party cloud-hosted LLM
[15](#rd-party-cloud-hosted-llm)](#rd-party-cloud-hosted-llm)

[3.1.2 Open-source/proprietary LLM
[15](#open-sourceproprietary-llm)](#open-sourceproprietary-llm)

[3.1.3 3<sup>rd</sup> party model vs open-source/proprietary
[15](#rd-party-model-vs-open-sourceproprietary)](#rd-party-model-vs-open-sourceproprietary)

[3.2 How to choose the right Foundation Model
[16](#how-to-choose-the-right-foundation-model)](#how-to-choose-the-right-foundation-model)

[4 Prompting strategies
[18](#prompting-strategies)](#prompting-strategies)

[5 Reasoning in LLMs [19](#reasoning-in-llms)](#reasoning-in-llms)

[5.1 What is reasoning for LLMs?
[19](#what-is-reasoning-for-llms)](#what-is-reasoning-for-llms)

[5.2 LLM reasoning techniques
[19](#llm-reasoning-techniques)](#llm-reasoning-techniques)

[5.2.1 Training-time techniques
[19](#training-time-techniques)](#training-time-techniques)

[5.2.2 Inference-time techniques
[20](#inference-time-techniques)](#inference-time-techniques)

[6 Retrieval-Augmented Generation (RAG)
[23](#retrieval-augmented-generation-rag)](#retrieval-augmented-generation-rag)

[7 Model Context Protocol (MCP)
[27](#model-context-protocol-mcp)](#model-context-protocol-mcp)

[7.1 MCP Architecture [27](#mcp-architecture)](#mcp-architecture)

[7.2 MCP vs API [29](#mcp-vs-api)](#mcp-vs-api)

[8 AI Agents [31](#ai-agents)](#ai-agents)

[8.1 What is an Agent? [31](#what-is-an-agent)](#what-is-an-agent)

[8.2 Some issues with AI Agents:
[31](#some-issues-with-ai-agents)](#some-issues-with-ai-agents)

[8.3 Context Engineering for AI Agents
[32](#context-engineering-for-ai-agents)](#context-engineering-for-ai-agents)

[8.4 Key-Value (KV) Cache Optimization
[33](#key-value-kv-cache-optimization)](#key-value-kv-cache-optimization)

[9 Other [35](#finetuning)](#finetuning)

[9.1 Quantization [35](#quantization)](#quantization)

[9.2 Small Language Models
[35](#small-language-models)](#small-language-models)

[9.3 Security [36](#security)](#security)

# What is AI Engineering?

AI Engineering is a new engineering discipline born as a consequence of
2 main phenomena:

- AI models got dramatically better at solving real problems

- The barrier to building AI models has gotten much lower

At its core, AI Engineering is about **building applications on top of
some *Foundation Models***, massive AI models already trained by big
tech companies.

Unlike Machine Learning Engineering, which focuses on training models
from scratch, AI Engineering leverages existing models, and focuses on
their adaptation & expansion.

<img src="media/image2.png" style="width:6.26597in;height:3.52014in" />

## AI machine analogy

<img src="media/image4.svg" style="width:6.26806in;height:3.52569in" />

# Large Language Models

## What is an LLM?

A Large Language Model (LLM) is a machine learning model trained to
predict the next token given a sequence of previous tokens. Tokens are
sub-word units (not always full words) produced by a tokenizer (more
details later).

*Example: an LLM predicts the next word of a popular meme, because the
model has been trained with examples that contain it.*  
Input: "sudo make me a"  
Output: "sandwich"

At its core, an LLM models the probability distribution:

P(next token \| previous tokens)

LLMs are a type of **Self-Supervised** models: instead of requiring
labelled data, these models learn by predicting parts of their input
data. This is a key factor that made LLMs possible, as labelling data is
a high human effort task.

LLMs have expanded to include not only text, but also images, audio, and
video, becoming **Large Multi-Modal Models**.

Today, LLMs (mainly Foundation Models) power a large set of use cases,
including coding assistants, image generation tools, customer support,
sophisticated data analysis.

## LLM Architecture

### Tokens

How does an LLM process input text?

An LLM does not process raw strings directly. Instead, the text is first
passed through a **tokenizer**, which converts the string into a
sequence of **tokens**.

**Tokens** are **discrete units** drawn from a **fixed vocabulary** of
**subword pieces** learned during model training. Tokens do not
necessarily correspond to full words: they may represent word fragments,
punctuation, whitespace, or even single characters. This allows the
model to efficiently represent both common words and rare or novel
strings.

Example:

Sentence: *All that glitters isn’t gold.*

A plausible tokenization (varies by model) might look like:

\["All", " that", " glitters", " isn", "’t", " gold", "."\]

or (for a more fragmented tokenizer):

\["All", " that", " glit", "ters", " is", "n", "’", "t", " gold", "."\]

**Why subword tokenization matters**

Breaking words into smaller subword units allows the model to learn
reusable linguistic patterns. For example, a suffix like **“-ing”**,
**“-ed”**, or **“-tion”** may appear across many different words. When
such fragments are represented as tokens, the model can more easily
generalize grammatical structure and meaning across the vocabulary,
rather than treating each word as an entirely separate symbol.

Finer-grained tokenization also improves robustness to rare, misspelled,
or previously unseen words, since they can be composed from known
subword pieces.

Every LLM has a vocabulary of tokens to tokenize the input it receives.

<figure>
<img src="media/image5.png" style="width:2.43095in;height:2.39216in" />
<figcaption><p>Figure 1: Example of a vocabulary of tokens, containing
50,000 tokens.</p></figcaption>
</figure>

> Tokenization converts raw text into a sequence of subword tokens from
> a fixed vocabulary, allowing LLMs to efficiently handle both common
> and rare text.

### Word embeddings

Tokens are still not suitable for a machine learning model, because
they’re just strings, and a machine learning model works with numbers.
For this reason, each token of the initial vocabulary is turned into a
**vector** of fixed length.

So, from the initial vocabulary, each token goes through a process of
**embedding**: it is transformed into a vector of fixed dimension, where
the value of each dimension is a number that *embeds* (sends, maps,
transforms) the token into a n-dimensional space.

<figure>
<img src="media/image6.png" style="width:4.83737in;height:2.00655in" />
<figcaption><p>Figure 2: Matrix containing all the vectors of the
vocabulary tokens</p></figcaption>
</figure>

<figure>
<img src="media/image7.png" style="width:4.0434in;height:2.92571in" />
<figcaption><p>Figure 3: Simplified visualization of tokens in the
embedding space (assuming the number of dimensions of the embedding
space is 3).</p></figcaption>
</figure>

**Q: How are the values of each dimension assigned?  **
A: They are parameters of the LLM, so they are learned from data. Once
they have been learnt, they represent a sort of weighted average of the
meanings of the given token.

**Q: What do the dimensions represent?  **
Practically, we can’t know, because the neural network filling the
values is a black box.  
To get an idea, we can think of dimensions as features of words, such as
gender (e.g., male/female), verb tense (e.g., present/past) or
*pertinence to a specific field* (e.g., finance, nature, IT, …), or
other features that could describe words.

<img src="media/image8.png" style="width:3.86927in;height:1.52405in" />

> The key result from embedding is that words with similar meaning end
> up into the same region of this n-dimensional space.

For example, vectors of tokens that refer to buildings (skyscraper,
tower, roof, …) are close to each other.

<img src="media/image9.png" style="width:4.81745in;height:2.3751in" />

**Q: How many dimensions are there?  **
A: Depends on the model. GPT3 has roughly 12,000.

Sources:

- <https://www.youtube.com/watch?v=wjZofJX0v4M&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=6&pp=iAQB>

- <https://www.youtube.com/watch?v=eMlx5fFNoYc&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=8>

### Attention & Transformer

Once we have vectors representing specific tokens, a new challenge
arises: in natural language, the same word can have different meaning
based on where it is used. For example, in English, the word *set* has
more than 400 different meanings. Thus, having one single vectorization
for the token *set* wouldn’t be very useful for training LLMs.

> Attention is an operation that correlates each token with a certain
> number of tokens preceding it. Previous tokens that result more
> correlated with the given token can be used to update the given
> token’s vector, pushing it more towards the sematic area that it
> belongs to.

<img src="media/image10.png" style="width:6.06575in;height:3.66458in" />

In LLMs, the attention operation is computed through matrices whose
values are learned by the LLM itself from data. Furthermore, many
attention operations are put in sequence into LLMs, to make the model
understand more context at each iteration, and this sequence of
attention operations is called a **Transformer** architecture.

<img src="media/image11.png" style="width:5.72963in;height:4.18518in" />

## LLM Development

Once we have the LLM architecture set up, we have billions of parameters
to be learned from data. The process of learning parameters and
furtherly optimizing the LLM can be referred as the LLM development, and
consists of the following stages.

1.  **Training data gathering**: massive text corpora coming from all
    the web (e.g., Wikipedia, Reddit – only posts with at least 3
    upvotes, etc.). **Note**: according to the Chinchilla Scaling Law,
    the number of training tokens should be about 20 times the model
    size – so, a 3B parameter model needs around 60B training tokens

2.  **Tokenization** of the input text

3.  **Self-supervised Pre-training**: for each text corpora (e.g., a
    Wikipedia page), use the first n tokens to predict the n+1th token,
    then compare with the real n+1th token and perform a backward pass,
    updating the LLM’s parameters.

4.  **Supervised Fine-Tuning (SFT)**: the pre-trained model (or *base*
    model) is fine-tuned on curated datasets of Q&A pairs to align
    outputs with desired formats and behaviours.

5.  **Reinforcement Learning from Human Feedback (RLHF)**: making the
    model generate multiple candidate answers to the same question, and
    then having a human rank the answers from best to worst. A reward
    model is then trained and used to further fine-tune the LLM.

6.  **Specialization and System-Level Enhancements**: specialize/enhance
    the model for a specific field/task. Example:

    - Conversational applications

    - Web search LLMs

    - Automations

    - Retrieval Augmented Generation (RAG)

    - Model Context Protocol (MCP)

    - Chatbots

    - Multimodal models (text, image, audio, video)

    - Code assistants

    - Reasoning models

    - Agents

    - Distilled, small and low-cost models

## LLM Evaluation

Evaluating an LLM is significantly harder than other ML models, for
several reasons:

- The problems solved by LLMs are more complex: to evaluate a
  mathematical proof, you need to have proper background and inspecting
  each step

- Tasks are often open-ended, with multiple correct answers

- For general purpose models, you need to evaluate not just known tasks,
  but explore many other possible capabilities

### Metrics to evaluate an LLM

| **Metric** | **Definition** | **Explanation** | **What a good score looks like** |
|----|----|----|----|
| **Cross-entropy (CE)** | Expected negative log-probability the model assigns to the true next token | Measures how surprised the model is by the ground-truth token. Lower CE means the model assigns higher probability to correct tokens. CE reflects mismatch between model distribution and data distribution. | **Lower is better**. The theoretical lower bound is the true entropy of the data distribution. |
| **Perplexity (PPL)** | Exponential of cross-entropy | Interpretable version of CE: the *effective number of equally likely tokens* the model considers at each step. | **Lower is better**. Must always be compared **on the same dataset, tokenizer, and vocab size**. |
| **Accuracy** | Fraction of exactly correct outputs | Works only for **closed-form, unambiguous tasks** (classification, multiple choice, exact answers). | Close to **100%** on deterministic tasks; meaningless for open-ended generation. |
| **Exact Match (EM)** | Binary match between model output and reference | Very strict; penalizes paraphrases and formatting differences. | High only when answers are standardized. |
| **Lexical overlap (BLEU / ROUGE)** | Token-level overlap between output and reference | Measures surface similarity; weak for reasoning and paraphrasing. | Task-dependent; best used for summarization or translation. |
| **Semantic similarity (BERTScore, cosine sim)** | Embedding-level similarity between output and reference | Captures paraphrases and meaning similarity better than lexical metrics. | Higher is better; thresholds are task-specific. |
| **Functional correctness** | Whether the output achieves the intended goal | Often used for **code, agents, and tools**: does the program run? does the task succeed? | Pass rate on unit tests or task completion. |
| **Hallucination rate** | Frequency of unsupported or fabricated claims | Measures whether outputs contain information not grounded in the input or known facts. | **As low as possible**; usually measured via human or model-based evaluation. |
| **Faithfulness / Groundedness** | Degree to which output is supported by provided context | Especially important in RAG systems; answers must be derivable from the given documents. | High entailment; minimal unsupported statements. |
| **Toxicity / Safety** | Presence of harmful, biased, or unsafe content | Often measured via classifiers or red-team prompts. | Below defined safety thresholds. |
| **Calibration (ECE)** | Alignment between confidence and correctness | A well-calibrated model is not overconfident when wrong. | Low Expected Calibration Error. |
| **Robustness** | Stability under perturbations | Measures sensitivity to prompt wording, noise, or adversarial inputs. | Minimal performance degradation. |
| **Latency / Throughput** | Time and tokens per second | Operational metric, not model quality per se. | Task- and system-dependent. |

### Other ways to evaluate LLMs

Use another AI as a judge, that can evaluate responses and justify their
evaluations, helping with transparency.

AI Judges can be used in 3 ways:

- Score outputs

- Compare outputs to references

- Pick the best of 2 responses

When using another LLM as a judge, endure the prompt contains:

- Evaluation task

- Criteria

- Scoring system

- Few shots examples<span id="_Toc218633902" class="anchor"></span>

## Current LLMs limitations

- **Distraction by irrelevant context:** the model tends to use all the
  information we give it, even if they are irrelevant

<img src="media/image12.jpeg"
style="width:3.58125in;height:1.03164in" />

- **Inability to self-correct**: if you ask the model to correct itself
  when it was correct, it happens that it changes its answer and
  generate a false one

<img src="media/image12.jpeg"
style="width:3.58053in;height:0.72785in" />

- **Sensitive to order of information we give the model**: it pays more
  attention to the **first** and **last** information we give in the
  prompt

<img src="media/image12.jpeg"
style="width:1.61343in;height:1.48101in" />

- **High cost of improvement**: going from a 3% error rate to 2% might
  need an order of magnitude more data, compute, or energy

- **Training data**: we will eventually run out of training data, and we
  will need to use AI-generated data to further train models, possibly
  causing performance degradation, or requiring access to
  copyrighted/proprietary data

- **Electricity**: data centres already consume 1-2% of the global
  electricity, limiting how much they can grow without significant
  impact on overall electricity

# How to build applications on top of a Foundational LLM

## LLM Application base architecture

There are 2 ways to leverage a LLM model to build applications:

1.  **Accessing a 3<sup>rd</sup> party cloud-hosted LLM** (e.g., GPT,
    Gemini, …)

2.  Using an **open-source/proprietary LLM**

### 3<sup>rd</sup> party cloud-hosted LLM 

The mechanism to access the model is *serverless* and looks like the
following. Having these components:

- **Application Server**: where your application is hosted and runs
  (e.g., ChatGPT)

- **LLM API**: an interface to interact (as a physical user or as an
  application) with an LLM (e.g., AWS Bedrock, OpenAI API)

- **LLM**: the actual model parameters & architecture

- **Inference service**: the machine that hosts the LLM and runs the LLM
  queries

the server sends a request to the LLM API, which triggers the LLM model.
Then, the API gets back the answer to your application server.

<img src="media/image13.png" style="width:4.97598in;height:2.74103in" />

### Open-source/proprietary LLM

In this option, you won’t need an API to call the LLM model.

### 3<sup>rd</sup> party model vs open-source/proprietary

- Data privacy

- Data lineage and copyright: you can’t know form where the training
  data come from

- Performance: strongest models are not open-source

- Lock-in & dependence on a vendor

## How to choose the right Foundation Model

With the increasing number of readily available foundation models,
selecting the right one for your application can be challenging.

The selection process involves 6 steps:

1.  **Filter out** models whose **hard attributes** don’t work for you:

    - License restrictions

    - Training data

    - Model size

    - Privacy requirements

    - Level of control needed

2.  Use **public benchmark data** to further narrow down the candidates
    list

<img src="media/image14.png" style="width:4.84691in;height:2.22584in" />

> **Note**: benchmarks may suffer from **data contamination**: they were
> evaluated on the same data used for training. To detect this, you can
> use **perplexity**: if it’s very low, it’s possible the model has
> already seen the data.

3.  Put down a framework to find the **best achievable performance on
    your task**, e.g.:

    - **Domain specific capabilities**: how well does the model
      understand the domain?

    - **General capabilities**: how coherent is the output

    - **Instruction-following capabilities**: does the model follow the
      expected format

    - **Cost & latency**: how expensive is running the model and how
      quickly does it respond

4.  **Run the evaluation**

    - Generate a set of test queries and generate for each multiple
      responses to inspect

    - Test on multiple clusters of queries (e.g., different topics): the
      evaluation result may suffer from the Simpson’s Paradox, i.e., the
      model works well on the aggregated queries, but poorly on a
      particular cluster

    - Bootstrap the same data to see if the model produces consistent
      output

5.  Check whether some **model’s performance** can be **improved** using
    **adaptation techniques** like **prompt engineering** or
    **fine-tuning**  
    Only **Soft attributes** can be improved:

    - Accuracy

    - Toxicity

    - Factual consistency

6.  **Map models on a cost vs performance plane** and select the model
    that gives the best performance for your budget

> Model selection remains one of the most challenging but important
> topics in AI Engineering: with the rapidly growing number of
> foundation models available, the challenge isn’t developing a model,
> but selecting the most suited for your use-case.

# Prompting strategies

- **Concatenate instructions** (Concat Principle): gather requirements
  into the first prompt. Avoid specifying corrections in many turns

- **Key info first & last** (U-Shape Attention): state main objective at
  the top. Restate critical constraints at the bottom

- **Filter noise**: remove irrelevant context or filler text. Summarize
  instead of pasting raw logs

- **Reset conversation if needed**: if model drifts, start new
  conversation and restate full instructions

- **Structure prompts**: use structured prompts. Break down into steps
  of **checklists, bullet points**

A common pattern to design a prompt is dividing it into 3 parts:

1.  **System prompt**: give a **role** to the LLM (e.g., “You are a
    conversational chatbot…”)

2.  **User prompt**: explanation of the **task** the model needs to
    perform (e.g., “Summarize the following article”)

3.  **Additional prompt**: add any other **information** that could be
    relevant to the model to produce the answer. Examples include
    various techniques:

    - chain-of-thought: provide examples

    - RAG: provide external documents/data

    - MCP: provide descriptions of tools that the model can call

<img src="media/image15.png" style="width:5.5019in;height:1.92439in" />

# Reasoning in LLMs

## What is reasoning for LLMs?

In the LLM context, **reasoning** refers to generating **intermediate
representations** (tokens or latent steps) that help the model arrive at
a correct final answer.

Indeed, most popular LLMs providers (e.g., OpenAI, Claude, Google)
expose 2 implicit modalities for how the model generates answers:

- **Normal (fast) mode**: the model answers without spending many tokens
  (i.e., it only considers the question provided by the user). Answers
  are quicker, suitable for simple queries.

- **Thinking (extended reasoning) mode**: the model is allowed to spend
  extra compute cycles, generating more tokens before giving the answer.
  Answers are slower, but accuracy is higher.

There are many ways to make a model reason.

## LLM reasoning techniques

Multiple techniques have been developed to make LLMs reason. They can be
distinguished into 2 main groups:

- **Training-time** reasoning techniques: train models from scratch with
  output that includes reasoning steps

- **Inference-time** reasoning techniques: make the model generate
  reasoning steps only when answering a question

### Training-time techniques

- **Chain-of-thought (CoT) training**: train the model from scratch with
  output text that contains **intermediate steps** with reasoning (e.g.,
  output = rationales + final solution).

  - This has been seen to improve accuracy of answers

  - Limitation: requires massive amounts of annotated rationales

- **CoT supervised Fine-Tuning (SFT)**: take a pre-trained LLM (e.g.,
  Chat GPT) and fine-tune it on a dataset of Q&A where answers include
  **step-by-step** reasoning+final answer

  - Dramatically improves accuracy on mathematics questions

  - Limits: costly to collect rationales; fine-tuning limited to domains
    represented in the reasoning dataset (e.g., mathematics), so poor
    generalization

- **Reinforcement Learning with Verifiable Rewards (RLVR)**

  - In the Reinforcement Learning phase of the model development,
    instead of using a reward given by human preferences, give a reward
    based on **accuracy** and **format**.

    - For **format**, use an LLM “judge” to evaluate adherence to
      expected format (e.g., place reasoning steps inside \<think\>
      tags)

    - For **accuracy**, use a compiler to verify coding answers and a
      deterministic system to evaluate mathematical answers

  - Used mostly for maths and programming problems

### Inference-time techniques

Instead of increasing training cost, **inference-time scaling**
increases computation *only when answering a question*. This enables
usage of smaller base models requiring **less pre-train compute** and
flexible cost–accuracy trade-offs, even though introducing **additional
test-time compute**.

- **Chain-of-Thought prompting**: provide **step-by-step** reasoning
  **in the prompt**, and the LLM will tend to imitate reasoning in the
  answer

<img src="media/image16.jpeg"
style="width:3.23267in;height:1.60362in" />

> There are 2 variants:

- **Few-shot CoT**: give examples Q&A pairs with reasoning steps.
  **Limitation**: not always the LLM gets the idea.

- **Zero-shot CoT**: add a cue to the prompt, e.g.: “let’s think
  step-by-step”

> CoT prompting is one of the simplest and most effective inference-time
> scaling methods.

- **Best-of-N Sampling**:

  1)  make the model generate **N independent answers** to the question

  2)  Use a verifier model (or heuristic) to **select the best** **one**

> Key idea in this technique is that **verification** is **simpler**
> than **generation**: the model doesn’t produce intermediate tokens.
> How is it made? With a second model trained to verify the
> answer<img src="media/image17.jpeg"
> style="width:4.81696in;height:1.80863in" />

- **Chain-of-Thought Decoding & Self-Consistency**

  - To understand this technique, we first need to understand how an LLM
    generates the next token

<img src="media/image18.jpeg"
style="width:2.56034in;height:2.65347in" />

- At each step, the model produces a probability distribution over
  tokens

- Then, a decoding strategy selects the next token:

  - **Greedy decoding**: take highest probability token

  - **Sampling with temperature**: randomly select the token according
    to the probabilities

- **Temperature**: if you select a high temperature (\>0), you make the
  probability distribution wider, leading to a more “creative” answer

> <img src="media/image19.png" style="width:3.52018in;height:2.63526in" />
>
> **Chain of Thought Decoding** involves the following steps:

- sample k alternative first tokens

- generate full reasoning paths from each

- aggregate final answers (e.g., with majority vote, selecting the final
  answer that appears most frequently)

> This can enable reasoning without additional training or special
> prompts<img src="media/image20.jpeg"
> style="width:4.62376in;height:1.37517in" />
>
> We can say that the model already contains the “reasoned” answer, we
> only need to make it generate it

- **Limitation**: self-consistency can only be applied to tasks where
  the final answer can be aggregated by exact match (e.g., a number for
  math problems)

- **Solution**: **Universal Self-Consistency**. Call the LLM itself to
  select the most consistent answer among all

- **Key difference with Best-of-N:** best of N generates all the
  answers, then selects the most consistent. CoT decoding generates k
  first tokens, then for each one computes the full answer. Then,
  selects the most consistent.

> Reasoning in LLM is quite expensive, because you generate more tokens.
> Use for complex math problems, logic, advanced coding, novel problems,
> strategic planning. Don’t use for simple queries, knowledge retrieval,
> real time applications (e.g., chatbot), or high-volume tasks.

# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is an architecture for GenAI
systems aimed at overcoming the limitations of static knowledge in LLMs.

RAG integrates a generative model with an **information retrieval**
component, that allows it to retrieve information from **external data
sources**, which are typically **indexed** using **vector embeddings**
and queried with **similarity search**.

The output of the model is therefore not based solely on parametric
inference, but on the **dynamic injection** of **selected informational
context**.

<img src="media/image21.png" style="width:3.96021in;height:2.85235in" />

RAG is composed by 2 components:

- **Retriever**: the system that fetches information

- **Generative model**: the LLM which uses the retrieved information to
  augment the user’s query

The retriever performs 2 key actions:

- **Indexing**: processing data so that it can be retrieved more quickly
  later

- **Querying**: sending a search query to retrieve data relevant to it

Given that the retrieved data will be attached to the prompt, the
retrieved data must not contain too many tokens, because otherwise it
would make the context grow bigger. This is why documents are first
split in **chunks**, and then only the relevant chunk is attached to the
query. Some key ideas of chunks:

- Overlapping chunks ensure that relevant context is fully contained in
  at least one chunk

- Smaller chunks: more information, because you can fit more into a
  prompt, but also less information

- Smaller chunks increase computational overhead

- There is no universal overlap percentage or chunk size. Determine
  empirically

- Add smaller additional chunks containing a summary of the document

The retriever can be implemented with several different algorithms:

- **Term-based:** finds relevant documents based on keywords. TF-IDF can
  be used to enhance results. Examples include Elasticsearch. Fast, but
  can retrieve documents not relevant.

- **Embedding-based**: compute relevance at a semantic level, rather
  than at a lexical one, ranking documents based on how well their
  meaning aligns with the query. More resource-demanding but gets more
  relevant data.

In **production** environments, a **combination of the two approaches**
is often used: use a term-based to narrow the search, then use the
embedding-based to find the most suitable.

From a computational point of view, the RAG flow is linear and
deterministic: a query is transformed into a vector, compared with a
semantic space, and the closest matching documents are **concatenated to
the user prompt**.

<img src="media/image22.png" style="width:5.03011in;height:3.15371in" />

The model performs a conditioned synthesis, improving **factual
accuracy**, **source traceability**, and **business domain alignment**.
This architecture is particularly effective in enterprise, regulated, or
knowledge-intensive contexts, where the **separation between model and
knowledge** also constitutes a governance control.

> RAG: Retrieve the context, Augment the query, Generate the response

Rag can also work with tabular data and image data.

<img src="media/image23.png" style="width:5.2072in;height:3.02936in" />

<img src="media/image24.png" style="width:4.87001in;height:2.86071in" />

<img src="media/image25.jpeg"
style="width:3.36898in;height:4.02777in" />

Sources:

- <https://medium.com/@tejpal.abhyuday/retrieval-augmented-generation-rag-from-basics-to-advanced-a2b068fd576c>

- https://medium.com/@mawatwalmanish1997/agentic-rag-7e6d7838e217

# Model Context Protocol (MCP)

We have seen how with RAG LLMs can expand their knowledge to information
contained in external data sources. But what if the information is not
ready to be pulled from a database, but needs to be retrieved by
performing some kind of action (e.g., fetch the price of a flight ticket
from the Airline website).

Model Context Protocol (MCP) is a **protocol** (i.e., a way to
communicate between two systems) designed to **standardize** how
**applications provide context** **to an LLM**.

MCP **standardizes connection** between **AI applications**, **LLM**,
and **external data sources**. It is like a USB port for your AI
application: it doesn’t matter what the peripheral is, USB works for
all.

## MCP Architecture

MCP involve 2 actors: a **client** and a **server**.

<img src="media/image26.png" style="width:4.27863in;height:2.53987in" />

- An **MCP host** can run several **MCP clients**. Each client can send
  requests to many MCP servers and connect to them through the **MCP
  Protocol**

- **MCP servers** expose **capabilities** that can be accessed by the
  MCP clients. MCP servers expose capabilities through *primitives*:

  - **Tools**: discrete actions or functions the AI can call (e.g.,
    getWeather, createEvent, …). The MCP server exposes the tool name,
    description, input/output schema, capabilities list. When an LLM
    uses an MCP client to invoke a tool, the MCP server executes the
    underlying function

  - **Resources**: read-only data or documents the server can provide
    and then the client can retrieve on-demand (e.g., database schemas)

  - **Prompt templates**: pre-defined templates providing suggested
    prompts

> MCP servers publish machine readable catalogs, such as *tools/list*,
> *resources/list*, *prompts/list*

- Many popular applications have developed their MCP server, such as
  Github, Google Maps, Spotify, etc. (e.g., a server to access a
  database, another to access a code repository, an email server, …)

<img src="media/image27.png" style="width:1.80347in;height:2.41999in" /><img src="media/image28.png" style="width:1.77515in;height:2.44083in" />

Example:

"Figure out if this email involves \[situation\]." which the app turns
into a MCP call for matching \[email address\] to \[case number\] from a
database and then takes \[case number\] and runs a query to get \[case
history\] then looks for \[situation\] in case history. Output: "yes" or
"no"

The flow at runtime looks like this:

1)  The user sends a query to the Client MCP

2)  The AI Agent can query the MCP server **at runtime** to discover
    what primitives are available, without having to redeploy code every
    time an MCP server changes

3)  The MCP forwards the query to an LLM

4)  The LLM has access to a set of tools that it can use; so, it
    identifies the most relevant tools to provide a more accurate answer
    to the query, and sends to the MCP servers a request to use these
    tools

5)  The MCP client connects with the MCP servers requested by the tools

6)  The MCP servers sends an answer to the MCP client

7)  The MCP client augments the initial prompt with the information
    retrieved from the external systems and sends it back to the LLM

8)  The LLM produces the final output

Why MCP instead of just putting tool descriptions in the prompt?

MCP standardizes tool discovery and invocation, reducing prompt size,
improving cache reuse, and enabling safer, inspectable agent behavior.
By moving tool schemas out of free-form prompts, MCP improves
determinism and system interoperability.

Source: <https://www.youtube.com/watch?v=7j1t3UZA1TY>

## MCP vs API

APIs are another way of letting one system access another system’s
functionality or data. An API defines a **set of rules** describing
**how to request information to a system**. By using an API, developers
can integrate capabilities of an external system in an easy way, instead
of reinventing the wheel evert time. For example, an e-commerce may use
a payment API to process credit card payments.

APIs act as an **abstraction layer**: the client doesn’t need to know
the internal details of the system it needs to invoke, the server. The
only thing the client needs to know is **how to format the request** and
**how to understand the responses** from the server.

There are many types of APIs, but one of the most ubiquitous is the
**RESTful API**. REST uses **http** protocol to communicate between
systems, which consists of standard http methods the client can use:

- **get** to retrieve data

- **post** to create data

- **put** to update data

- **delete** to remove data.

For example, a REST API for a library system might have an **endpoint**
such as *GET/books/123* to fetch book details, or *POST/loans* to borrow
a book. **Each endpoint returns data**, often in a JSON format,
representing the result.

Many foundation LLMs are offered over REST: you send a JSON prompt, you
receive a JSON with the response.

**API vs MCP**

<table style="width:100%;">
<colgroup>
<col style="width: 33%" />
<col style="width: 33%" />
<col style="width: 33%" />
</colgroup>
<thead>
<tr>
<th></th>
<th>REST API</th>
<th>MCP</th>
</tr>
</thead>
<tbody>
<tr>
<td>Model architecture</td>
<td>Client/Server: client sends http requests (get, post, …) to a
server, then the server responses with a JSON</td>
<td>Client/Server: the MCP client sends a request (tools calls) to a MCP
server and gets a response</td>
</tr>
<tr>
<td>Abstraction layer</td>
<td colspan="2">Hides implementation details</td>
</tr>
<tr>
<td>Integration between systems</td>
<td colspan="2">Simplified, letting developers wire systems together
without reinventing the wheel</td>
</tr>
<tr>
<td></td>
<td>General purpose: not created specifically for LLMs</td>
<td>Purpose built: specifically designed to integrate LLM apps with
external data and tools in ways that align with how AI agents
operate</td>
</tr>
<tr>
<td>Sensibility to changes</td>
<td>If the API changes, client needs to be updated and re-deployed</td>
<td><strong>Dynamic self-discovery</strong>: an MCP client can ask an
MCP server “what can you do?” and get in return a description of all
available functions and data the server offers</td>
</tr>
<tr>
<td>Standardization of interface</td>
<td>Each API is unique: specific endpoints, authentication schemes vary
between services</td>
<td>Every server speaks the same protocols. All MCP servers respond to
the same calls</td>
</tr>
</tbody>
</table>

**Note**: often, MCP servers are implemented as wrappers of an existing
API. So that developers can reuse the native services exposed by the
API. In this case, MCP are translations of APIs which provide a more
AI-friendly interface on top.

> While APIs are interfaces meant to utilised by other applications, MCP
> are interfaces/wrappers around APIs meant to be utilized by LLMs.

# AI Agents

## What is an Agent?

An Agent is a system designed to manage multi-step processes leveraging
LLMs.

An agent can:

- Be queried using an API

- Query an LLM

- Query external systems

- Interact with other agents

<img src="media/image29.png" style="width:4.93885in;height:2.77804in" />

<img src="media/image30.png" style="width:4.91729in;height:2.76591in" />

## Some issues with AI Agents:

- **Compound error effect**: When an Agent uses LLM calls multiple
  times, the error compounds. In a multi-step workflow, error
  probability increases very fast. In an agent has 95% probability of
  success at each step (very optimistic assumption), after 10 steps the
  probability of success is at 50%. This is a problem is related to how
  we build the workflow, rather than the LLM model itself. If the
  workflow includes steps taking as input the output of an LLM, the
  error compounds.

**Mitigations**:

- Use few independent, verifiable operations

- Use explicit rollback points

- Human confirmation gates

<!-- -->

- **Cost explosion**: in a multi-step agent, the model typically selects
  an **action**, and based on the selected action, chooses a tool. Then,
  that tool may give an answer that we feed back in to the model. This
  creates the need for the model to memorize history, i.e., increase the
  **context window**.

  - Context windows create quadratic token costs

<img src="media/image31.jpeg"
style="width:4.27104in;height:3.02497in" />

- Long conversations become prohibitively expensive at scale, so
  difficult to bring to production

- Agents succeed when they are **completely stateless (or *idempotent,
  ed.)***:

  - Description 🡪 function 🡪 done.

  - No context to maintain, no conversation to track, no quadratic cost
    explosion

  - Not “chat with your code” experience; rather, **focused tool** that
    **solves a specific problem** efficiently

- Most successful agents in production aren’t conversational: they’re
  bounded tools that do one thing well and then get out of the way

## Context Engineering for AI Agents

In designing production grade agent systems, AI is only 30% of the
problem. The other 70% is a problem of software engineering:

- **Design tools input/output carefully:** don’t overwhelm the context
  window. Use short sentences to communicate to agents if a tool’s
  operation was successful, without burning tokens

- **Abstract results**: a database might return 10,000 rows, but the
  agent only needs to know “query succeeded. 10k results. Here are the
  first 5.”, instead of all the raw data or execution log

- **Return structured, machine-readable exceptions/errors**: if a tool
  fails, what information does the agent need to retrieve from the tool
  API? Too little and it’s stuck; too much and you waste context. Each
  tool should return structured feedback that the agent can easily
  consume to make decisions. Don’t just return raw API responses.

- **Build APIs for machines, not humans:** APIs should be built as
  machine interfaces, not as if the agent were a person. The risk is
  agents that technically make successful API calls but can’t accomplish
  complex workflows because they don’t understand what happened.

- Design the agent to **execute verifiable, transparent, inspectable
  operations**

- Insert **human decision points** at critical junctions

- **Limit context**: use a sliding window to only maintain in memory
  **the last n messages**, and **summarize** all the previous ones in 5
  sentences

## Key-Value (KV) Cache Optimization

When an LLM produces an answer, to generate each new token, it needs to
make a forward pass of all the previous tokens. This is highly
inefficient. Another example: some agents operate through iterative
processes:

- Get user input

- Perform a chain of tool uses. For each tool:

  - select an action

  - execute it

  - produce an observation

  - append result to context

**Key-value cache** employs memory in place instead of performing all
these forward passes. The model stores in cache the key and value of all
previous tokens. If you use Claude Sonnet, generating tokens using
key-value cache will cost 0.30\$/MTok (Millions of tokens) vs
3.00\$/MTok to generate new tokens.

Best practices to get the max out of Key-Value cache:

- Maintain the beginning of the prompt fixed

- Avoid timestamps (even a single token can invalidate the cache from
  that token onward) and other dynamic elements

- Context should be **append-only** to avoid modification of previous
  actions or observations, invalidating the cache

- Make sure JSON serialization preserves the order of keys

  - {id: 1, name: x} is a different string from {name: x, id: 1}

- **Avoid dynamically adding or removing tools mid-iteration**, cause
  tools typically live near the front of the context. Changing them will
  invalidate the cache for all subsequent actions and observations. A
  solution is decreasing the probability of selecting a tool, instead of
  removing it (or increasing it, instead of adding it)

- **Avoid over-reliance on few-shot prompting**: if you put too many
  examples in the context, the model may tend to imitate the patterns
  even when it’s sub-optimal. Solve by introducing diversity, such as
  small variation in actions and observations

> Context window and inference strategy engineering are now as important
> as model choice itself.

<img src="media/image32.png" style="width:4.88661in;height:3.19205in" />

# Finetuning

Finetuning is the process of **adapting** a **model** to a **specific
task** by **further training** it and **adjusting** its **weights**.

Finetuning can improve a model’s performance in 2 ways:

1.  Enhancing domain-specific capabilities, e.g., coding

2.  Improving instruction-following capabilities, e.g., adapting to a
    specific output format

However, **finetuning requires a lot of memory**, often more than that
of a single GPU, making it expensive. This is why techniques for
reducing memory requirements of finetuning have been developed.

When to finetune?

- You’ve exhausted what you can achieve with prompt-based methods (RAG,
  prompt engineering)

- You need to produce consistent structured output

- You’re using a smaller model that needs to perform better at specific
  tasks

When not to finetune

- You need a **general-purpose** model: it can improve performance on a
  topic, but decrease it on other topics

- Pre-mature optimization: you still haven’t tried simpler approaches

Finetuning requires high computational resources because it involves
performing the **backward pass** to adjust weights of the neural network
behind the model.

The main contributors to the model’s memory footprint during finetuning
are:

- Total number of **parameters** of the model, which can be:

  - **Trainable parameters**: updatable during finetuning

  - **Frozen parameters**: fixed and not updatable after pre-training
    has concluded

*🡪The more the parameters, the more expensive the forward pass*

- Total number of **trainable parameters**

*🡪The more the trainable parameters, the more expensive the backward
pass*

- **Numerical representation** of the parameters

*🡪 The more bits used to store parameters, the more expensive the
storage of the model*

<img src="media/image33.png" style="width:4.30151in;height:2.18745in" />

## Model distillation

Finetuning a small model to imitate a larger model, using data generated
by the large model

<img src="media/image34.png" style="width:4.2057in;height:2.66899in" />

## Finetuning vs RAG

After having maximized performance gains from prompting, choosing
between finetuning and RAG depends on whether your model’s failures are
**information-based** or **behaviour-based**:

- If the model fails because it lacks information (e.g., private company
  data), RAG gives the model access to that information

- If the model fails to adhere to expected standards (e.g., output in
  the wrong format, or output correct but noisy/irrelevant), finetuning
  teaches the model to follow standards

- If the model has both problems, start with RAG as it’s simpler (e.g.,
  term-based solution) and then evolve. Then, use a combination of the
  two methods.

## Parameter-Efficient Finetuning (PEFT)

As finetuning all parameters requires lots of computational resources
and data, **partial finetuning** emerged. Partial finetuning focuses
only on specific layers (e.g., the last layer), reducing memory
requirements.

Parameter-efficient finetuning insert **additional weights** in the
model in strategic locations, to maximize the performance improvement
coming from finetuning and **minimizing required data** and **computing
resources** to perform it.

PEFT models fall into 2 categories:

- **Adapter-based methods** (additive methods): add new model weights

- **Soft prompt-based methods**: introduce special trainable tokens

> Full finetuning typically requires thousands to millions of examples.
> PEFT can work with hundreds.

## Multi-task finetuning

If you want to finetune the same model for different tasks, you have
different options:

- **Simultaneous finetuning**: give the model examples of the various
  tasks all at once (requires more data)

- **Sequential finetuning**: train first on task A, then on task B
  (requires less data but can lead to forgetting the first task)

- **Model merging**: finetune for different tasks separately, then
  “combine” the resulting models (this is different from **ensembling,**
  which combines multiple outputs; this technique combines **models**
  themselves) – but **how**?

  - **Summing**: sum the parameters of the models

  - **Layer stacking**: take different layers from different models

  - **Concatenation**: combine the parameters (not recommended, doesn’t
    reduce memory)

<img src="media/image35.png" style="width:4.07034in;height:2.12581in" />

## Finetuning workflow

### Development path

1.  Test your finetuning code on a small, cheap model to see it works

2.  Test your finetuning data on a mid-sized model

    - training loss should decrease with more data

3.  Run experiments with your target model to see how far you can push
    performance

4.  Map the **price/performance frontier** and select the model that
    best suits your budget

### Distillation path

1.  Start with finetuning the **strongest model** you can afford with a
    **small dataset**

2.  Train the best possible model with this small dataset

3.  Use this **finetuned model** to **generate more training data**

4.  Use the **expanded dataset** to **train** a **cheaper model**

# Adapting a model to a task

Example workflow to adapt a model to a specific task:

1.  Design evaluation criteria & evaluation pipeline

2.  Try **prompting** strategies to improve performance

3.  If performance doesn’t improve, try **RAG**, first simple (i.e.,
    keyword), then more complex (i.e., embeddings)

4.  If the model has behavioural issues, try **finetuning**

5.  Finally, **combine** the two methods for a performance boost

<img src="media/image36.png" style="width:4.20924in;height:2.70807in" />

# Other

## Quantization

LLMs weights are usually stored as 32-bit numbers. Converting the
storage from 32-bit to 8-bits can reduce the model size, introducing a
75% memory reduction (although this isn’t accurate, as quantized weights
are only the weights of the feed-forward neural networks contained in
the LLM, whereas the weights of the transformer aren’t quantized).

<img src="media/image37.png" style="width:5.21842in;height:2.58897in" />

> Quantization is applied at the end of training, so when weights have
> been already learned by the LLM. Thus, quantization help reduce the
> LLM’s storage cost and inference cost, but not the training cost.

##  Small Language Models

Large Language models have 3 to 300 billion parameters, which make their
training very resource hungry.

Small Language Models are simply models with less parameters, usually in
the range of 3 to 300 million, and can be trained with less resources.
These models can work reasonably well for limited domains, e.g. training
on a small/medium company files.

SLM are trained by companies on their proprietary data.

Benefits of a SML include:

- Easier to host

- Less costly to train

- Faster at inference

The process to train SLM is known as **Distillation**:

- Pass the same input to both an LLM, the “teacher”, and a SLM, the
  “student”

- Compare the output of the two models

- Perform a backward pass in the SLM, updating its parameters to reflect
  the adjustment introduced by the difference with the LLM output

> SLM try to condense the complex relations (expressed by the weights)
> that an LLM has learned, into the lesser weights of a SLM.

Sources

- AI Engineering: Building Applications with Foundation Models Paperback
  – 20 Dec. 2024, by Chip Huyen

- Marina Wyss Youtube Channel: AI Engineering in 76 Minutes (Complete
  Course/Speedrun!)

## Security

Use **anomaly detection** tools to identify **suspicious prompts**.

# Bonus

Interesting questions

**How can an LLM with billions, trillions of parameters not overfit?**

We don’t know if the model is generating text in or out of the training
data, so for all we know a model could just be **pattern matching** the
training data. It’s arguable that we don’t care here if the model
overfits.

**Does an LLM learn the etymology of words in different languages?**

Take a model that has been trained on billions of articles, forums,
books, etc. Is it possible that it can understand the etymology of words
across different languages?

They give the example of “tramonto” (“sunset”), which in Italian, as far
as I understand, derives from *trans* and *monti* (I might be wrong here
— you’re the expert), so it refers to the semantic field of mountains,
whereas in French you say something like *coucher de soleil*, which
would be “the sun going to sleep”, so it refers more to the semantic
field of sleeping, and in English “sunset”, yet another meaning.

And they ask: in different languages, does the concept of sunset — or
rather the vector that represents the word “tramonto” in embedding space
— point in different directions depending on the different etymologies?
For example, in Italian is the vector associated more with a “mountains”
direction, in French with “sleeping”, etc.?

The answer is no: LLMs do not learn the “latent” etymologies of words.

In the initial tokenization, the vector for “tramonto” does not yet have
a precise meaning: it is just a rough representation based on how that
word appears in texts, a kind of weighted average of the uses of
“tramonto” (“sunset”, “the sunset of something”, “sunset as fading
away”, …). It does not contain the etymology of the word, but only
statistical directions. At this stage, the model has no reason to
associate “tramonto” with mountains (except insofar as it has seen these
two concepts associated in the training data).

Here, however, the LLM is only at the beginning of the long chain of
operations that leads it to “understand” a word. The LLM begins to
“understand” meaning in what happens next, namely the attention
mechanism, which is an operation that updates the initial token based on
the tokens around it.  
Example:

“the sunset of an era” → the token “sunset” aligns more with concepts
like “end”, “decline”, and loses part of its link to the concept of the
sun.

“watching the sunset over the sea” → here “sunset” aligns with concepts
like “horizon”, “light”, “sky”.

Why? Because statistically, in the texts used to train the model (e.g. a
Wikipedia article), when “the sunset of an era” appears, sooner or later
in the same text words like “end”, “decline” will appear. The same goes
for sunset in the sense of the sun. In this way, the model associates
these words as being similar.

In different languages, the model simply learns to project this vector
into compatible semantic regions based on what it sees in billions of
training sentences (metaphors, idiomatic expressions, …). But in the
end, regardless of the language the model is trained on, it will always
learn to associate words that usually appear together, independently of
their etymology.

No, I’m not saying this.  
If in a language the concept of sunset is associated with metaphors or
idioms different from Italian, the LLM will learn them. What it will not
learn is something else: the “hidden” etymological roots inside words.

Take “sunset” and suppose it is tokenized as “sun” + “set”. Every time
the model encounters “sunset” in the training data, it records that
those tokens appear together, and therefore “sun” and “set” will end up
being connected in some way.  
But in the rest of the training data, the token “sun” appears much more
often in contexts like “hot”, “clouds”, “sky”, etc. So, at the end of
training, “sun” is pushed mainly toward those concepts, not toward
“set”.

In other words, etymology is overridden by the real and dominant usage
of the word in texts.

That said, a model can learn different meanings for the same concept in
different languages. It is enough that there are sufficient examples in
the training dataset.

Paradoxically, if we trained a model on a series of prefabricated texts
where the words “ciao” and “tutti” always and only appear as a pair, the
model would learn that these words are always used together, even though
this is not true in any language.
