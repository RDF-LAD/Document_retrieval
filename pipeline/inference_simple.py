import os
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from byaldi import RAGMultiModalModel
import json
from qwen_vl_utils import process_vision_info

class MultimodalRAGInference:
    def __init__(self, index_name: str, vlm_model_id: str ="Qwen/Qwen2-VL-2B-Instruct",th=0.00):
        print(f"Initialisation du pipeline RAG complet ---")
        
        print(f"Chargement de l'index ColPali : {index_name}")
        self.retriever = RAGMultiModalModel.from_index(index_name)

        print(f"Chargement du VLM de génération ({vlm_model_id})...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.th = th

# 1.  quantification 4-bit
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            bnb_4bit_quant_type="nf4"
        )
        

        self.processor = AutoProcessor.from_pretrained(
    vlm_model_id,
    min_pixels=256 * 28 * 28,
    max_pixels=640 * 28 * 28, 
)
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            vlm_model_id,
            quantization_config=quantization_config,
            device_map={"": 0}, 
            torch_dtype=torch.bfloat16
        )
            
        
        print("--- Pipeline prêt ---")

    def answer_query(self, query: str, k: int = 2) -> str:
        """
        Exécute la recherche des k documents les plus pertinents, 
        puis génère une réponse textuelle basée sur ces images et la query.
        """
        print(f"\n Requête : '{query}'")
            #load mapping
        with open("evaluation/mapping_id_name.json", "r", encoding="utf-8") as f:
            doc_id_to_name = json.load(f)
        # Étape A : Retrieval 
        results = self.retriever.search(query, k=k)
        retrieved_images = []

        # count the amount of tokens in the query : 
        colpali_processor = self.retriever.model.processor  
        batch = colpali_processor.process_queries([query])
        n_query_vectors = int(batch["attention_mask"].sum())
        
        print(f"Pages extraites par ColPali (Top-{k}) :")
        retrieved_pages =[doc_id_to_name.get(str(res.doc_id)) for res in results if res.score/n_query_vectors > self.th]
        retrieved_images = [Image.open(f"data/{doc_name}.jpg") for doc_name in retrieved_pages if doc_name]
        for  i,doc_name in  enumerate(retrieved_pages):
            print(f"   Nom du fichier : {doc_name}, Score_top {i+1}  : {results[i].score/n_query_vectors:.4f}")

        if not retrieved_images:
            return "Aucun document pertinent n'a été trouvé pour répondre à cette question."

        # Étape B : Construction du prompt multimodal pour le VLM
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in retrieved_images],
                    {
                        "type": "text", 
                        "text": f"Using only the visual materials provided above, answer the following question in detail and in technical terms: {query}"
                    }
                ]
            }
        ]

        # Préparation des inputs via le processeur du VLM
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages) 
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            #padding=True,
            return_tensors="pt"
        ).to(self.device)

        # Étape C : Génération de la réponse textuelle
        print(" Génération de la réponse par le VLM...")
        if self.device == "cuda":
            torch.cuda.empty_cache()  
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
        
        # Décodage de la sortie (en ignorant les tokens d'entrée)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text

if __name__ == "__main__":
    # Test rapide du module d'inférence
    rag_system = MultimodalRAGInference(index_name="colpali_index")
    
    query_test = "When was the Eiffel Tower built?"
    
    response = rag_system.answer_query(query_test, k=1) # au dessu de k= 3 trop de tokens d'entrée pour le modèle
    
    print("Réponse générée ---")
    print(response)