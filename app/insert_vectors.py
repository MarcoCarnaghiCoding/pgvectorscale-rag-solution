from datetime import datetime

import pandas as pd
from database.vector_store import VectorStore
from timescale_vector.client import uuid_from_time

from sentence_transformers import SentenceTransformer
import logging
import time
from typing import Any, List, Optional, Tuple, Union

# Initialize VectorStore
vec = VectorStore()

# Read the CSV file
df = pd.read_csv( '../data/Diccionario_Conceptos.csv', delimiter=';')
# Add the new column 'categoria' with the same value for all rows
df['categoria'] = 'Diccionario de Conceptos'



def get_embedding_model():
  model = SentenceTransformer('all-MiniLM-L6-v2')
  return model

def get_embedding(text: str) -> List[float]:
    """
    Generate embedding for the given text.

    Args:
        text: The input text to generate an embedding for.

    Returns:
        A list of floats representing the embedding.
    """
    text = text.replace("\n", " ")
    start_time = time.time()
    model = get_embedding_model()
    embedding = model.encode(text)
    elapsed_time = time.time() - start_time
    logging.info(f"Embedding generated in {elapsed_time:.3f} seconds")
    return embedding

# Prepare data for insertion
def prepare_record(row):
    """Prepare a record for insertion into the vector store.

    This function creates a record with a UUID version 1 as the ID, which captures
    the current time or a specified time.

    Note:
        - By default, this function uses the current time for the UUID.
        - To use a specific time:
          1. Import the datetime module.
          2. Create a datetime object for your desired time.
          3. Use uuid_from_time(your_datetime) instead of uuid_from_time(datetime.now()).

        Example:
            from datetime import datetime
            specific_time = datetime(2023, 1, 1, 12, 0, 0)
            id = str(uuid_from_time(specific_time))

        This is useful when your content already has an associated datetime.
    """
    content = f"Concepto: {row['concepto ']}\definicion: {row[' definicion ']}\nEjemplo: {row[' ejemplo']}"
    #embedding = vec.get_embedding(content)
    embedding = get_embedding(content)
    return pd.Series(
        {
            "id": 'DicConceptos' + str(uuid_from_time(datetime.now())),
            "metadata": {
                "category": row["categoria"],
                "created_at": datetime.now().isoformat(),
            },
            "contents": content,
            "embedding": embedding,
        }
    )


records_df = df.apply(prepare_record, axis=1)

# Create tables and insert data
#vec.create_tables()
#vec.create_index()  # DiskAnnIndex
#vec.upsert(records_df)
