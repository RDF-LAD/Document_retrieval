import os 
from byaldi import RAGMultiModalModel


def main():
    IMAGE_DIR = os.path.abspath("data")
    INDEX_NAME = "colpali_index"
    
    print(f" Ingestion of the corpus")
    print(f"Source folder : {IMAGE_DIR}")
    
    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(f"{IMAGE_DIR} unknown")


    print("ColPali loading")
    model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")

    print("Indexation ")
    model.index(
        input_path=IMAGE_DIR,
        index_name=INDEX_NAME,
        store_collection_with_index=True,
        overwrite=True
    )

    print(f"Indexation done")

if __name__ == "__main__":
    main()