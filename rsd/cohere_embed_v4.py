import os
from openai import AzureOpenAI

from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import backoff
import requests

load_dotenv(override=True)


def backoff_handler(details):
    """
    Custom handler for backoff events to log messages with extra attributes.
    """
    message = (
        f"Backing off {details['target'].__name__} "
        f"(args={details['args']}, kwargs={details['kwargs']}) "
        f"for {details['wait']:.1f}s "
        f"(tries={details['tries']}, elapsed={details['elapsed']:.1f}s, "
        f"exception: {details['exception']})"
    )


def give_up_handler(details):
    message = (
        f"Max retries reached for {details['target'].__name__} "
        f"(args={details['args']}, kwargs={details['kwargs']}, "
        f"elapsed={details['elapsed']:.1f}s, exception: {details['exception']})"
    )

def get_client():
    client = AzureOpenAI(
        api_version=os.environ['AZURE_OPENAI_API_VERSION'],
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        api_key=os.environ['AZURE_OPENAI_API_KEY']
    )
    
    return client


@backoff.on_exception(
    backoff.constant,
    (requests.exceptions.Timeout, requests.exceptions.ConnectionError), # Only retry for RetryException
    interval=60,
    max_tries=25,
    jitter=None,
    on_backoff=backoff_handler,
    on_giveup=give_up_handler,
)
def calculate_similarity(client, model_name, sentences, dimensions=1536):

    response = client.embeddings.create(
        input=sentences,
        model=model_name,
        dimensions=dimensions,
        encoding_format="float",
        extra_body={"truncate": "START"},
        extra_headers={"extra-parameters": "pass-through"}
    )
    
    embeddings = []

    for item in response.data:
        embeddings.append(item.embedding)

    # Calculate cosine similarity for all possible pairs
    cosine_similarities = cosine_similarity(embeddings)
        
    num_items = cosine_similarities.shape[0]
    sum_of_rescaled_similarity = 0.0
    num_pairs = 0
    
    for i in range(num_items):
        for j in range(i+1, num_items):

            sim = cosine_similarities[i][j]
            rescaled_sim = (sim + 1) / 2
            sum_of_rescaled_similarity += rescaled_sim
            num_pairs += 1

    # Calculate the average similarity
    average_similarity = sum_of_rescaled_similarity / num_pairs if num_pairs > 0 else 0.0
    
    return average_similarity, cosine_similarities
    

if __name__ == "__main__":
    model_name = "embed-v-4-0"
    client = get_client()
    sentences = [
        "The duck crossed the road",
        "The chicken crossed the road",
        "The road was crossed by the duck",
        "The duck did not cross the road",
        "The duck quickly ran fluttering its small wings and then crossed the road"*10000,
    ]
    similarity, cosine_similarities = calculate_similarity(client, model_name, sentences)
    
    print(f"Similarity: {similarity}")