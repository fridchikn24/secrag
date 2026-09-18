import time
import openai
from openai import OpenAI

def embed_texts(texts, model="text-embedding-3-small", batch_size=40):
    client = OpenAI()
    all_embeddings = []
    
    # Process texts in chunks of 'batch_size'
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        
        # Exponential backoff parameters
        retries = 5
        delay = 2.0  # Initial wait time in seconds
        
        while retries > 0:
            try:
                response = client.embeddings.create(
                    model=model,
                    input=batch
                )
                # Extract and store embeddings
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                break  # Success! Break out of the retry loop
                
            except openai.RateLimitError as e:
                retries -= 1
                if retries == 0:
                    print(f"\n❌ Permanent Rate Limit Failure on batch index {i}.")
                    raise e
                
                print(f"\n⚠️ Rate limit hit. Retrying batch in {delay:.2f}s... ({retries} retries left)")
                time.sleep(delay)
                delay *= 2.0  # Double the wait time for the next attempt
                
            except Exception as e:
                print(f"\n❌ Unexpected error occurred processing batch: {e}")
                raise e
                
        
        time.sleep(0.5)
        
        return all_embeddings
