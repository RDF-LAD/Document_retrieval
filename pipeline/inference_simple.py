import os
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from byaldi import RAGMultiModalModel
import json
from qwen_vl_utils import process_vision_info

class MultimodalRAGInference:
    def __init__(self, index_name: str, vlm_model_id: str ="Qwen/Qwen2-VL-2B-Instruct",th=11.35):
        print(f"Initialisation du pipeline RAG complet ---")
        
        print(f"Chargement de l'index ColPali : {index_name}")
        self.retriever = RAGMultiModalModel.from_index(index_name)

        print(f"Chargement du VLM de génération ({vlm_model_id})...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.th = th

# 1. Configuration de la quantification 4-bit
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            bnb_4bit_quant_type="nf4"
        )
        
        # Qwen2-VL utilise une résolution dynamique : sans limite, une page haute résolution
        # peut produire des dizaines de milliers de patchs. 1 token visuel = 28x28 pixels.
        # self.processor = AutoProcessor.from_pretrained(
        #     vlm_model_id,
        #     size={
        #         "shortest_edge": 256 * 28 * 28,  # équivalent de min_pixels
        #         "longest_edge": 640 * 28 * 28,   # équivalent de max_pixels (~640 tokens visuels max par page)
        #     },
        # )
        self.processor = AutoProcessor.from_pretrained(
    vlm_model_id,
    min_pixels=256 * 28 * 28,
    max_pixels=640 * 28 * 28,  # ~640 tokens visuels max par page
)
        
        # 2. Chargement du modèle avec un mapping strict sur cuda:0 pour éviter les splits CPU/GPU
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            vlm_model_id,
            quantization_config=quantization_config,
            device_map={"": 0}, # Force tout le modèle sur le premier GPU (index 0)
            torch_dtype=torch.bfloat16
        )
            
        
        print("--- ✅ Pipeline prêt ---")

    def answer_query(self, query: str, k: int = 2) -> str:
        """
        Exécute la recherche des k documents les plus pertinents, 
        puis génère une réponse textuelle basée sur ces images et la query.
        """
        print(f"\n🔍 Requête : '{query}'")
            #load mapping
        with open("evaluation/mapping_id_name.json", "r", encoding="utf-8") as f:
            doc_id_to_name = json.load(f)
        # Étape A : Retrieval (Recherche des pages .jpg pertinentes)
        results = self.retriever.search(query, k=k)
        retrieved_images = []
        
        print(f"Pages extraites par ColPali (Top-{k}) :")
        retrieved_pages =[doc_id_to_name.get(str(res.doc_id)) for res in results if res.score > self.th]
        retrieved_images = [Image.open(f"data/{doc_name}.jpg") for doc_name in retrieved_pages if doc_name]
        for  doc_name in  retrieved_pages:
            print(f"   Nom du fichier : {doc_name}")

        if not retrieved_images:
            return "Aucun document pertinent n'a été trouvé pour répondre à cette question."

        # Étape B : Construction du prompt multimodal pour le VLM
        # On formate le prompt pour injecter les images et la question
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
        image_inputs, video_inputs = process_vision_info(messages) # Fonction utilitaire standard Qwen-VL
        
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
            torch.cuda.empty_cache()  # libère la mémoire en cache laissée par ColPali
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
    
    query_test = "When creating a Cassandra keyspace, what was the parameter used to sort the table in ascending order?"
    
    response = rag_system.answer_query(query_test, k=1) # je dois utiliser k=2 car sinon j'ai trop de tokens d'input dans le modèle
    
    print("Réponse générée ---")
    print(response)