import os
import json
from byaldi import RAGMultiModalModel
import gzip


def main():
    INDEX_NAME = "colpali_index"
    BENCHMARK_PATH = "evaluation/benchmark.json" # Adaptez selon votre structure
    
    print(f"Chargement de l'index '{INDEX_NAME}' ---")

    model = RAGMultiModalModel.from_index(INDEX_NAME)

    #load mapping
    with open("evaluation/mapping_id_name.json", "r", encoding="utf-8") as f:
        doc_id_to_name = json.load(f)

    #load benchmark
    if os.path.exists(BENCHMARK_PATH):
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            benchmark = json.load(f)
        print(f"Benchmark found : {len(benchmark)} questions")
    else:
        print("benchmark not found.")
        return 0

    # 2. Boucle d'évaluation
    hits = 0
    k = 3 # On regarde dans le Top-3

    print(f"\n--- Évaluation du Retrieval (Top-{k}) ---")
    for item in benchmark:
        query = item["question"]
        target = item.get("docs", "")
        
        # late-interaction via ColPali
        results = model.search(query, k=k)
        #

        # IDs of retrieved pages.
        retrieved_pages =[doc_id_to_name.get(str(res.doc_id)) for res in results]
        
    
  
        success = target in retrieved_pages if target else True
        if success:
            hits += 1
            
        print(f"Question : {query}")
        print(f"  -> Targeted : {target}")
        print(f"  -> Found (Top-{k}) : {retrieved_pages}")
        for res in results:
            print(f"     - Doc ID : {res.doc_id}, Score : {res.score:.4f}")
        print(f"  -> Status : {' Success' if success else ' Failure'}\n")

    if benchmark:
        accuracy = (hits / len(benchmark)) * 100
        print(f" Global Retrieval Score (Hit@{k}) : {accuracy:.1f}% ---")

if __name__ == "__main__":
    main()