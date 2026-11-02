import os
import re
import warnings
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from byaldi import RAGMultiModalModel
import json
from qwen_vl_utils import process_vision_info
import time

# Avertissement de dépréciation sans conséquence sur min_pixels/max_pixels
warnings.filterwarnings(
    "ignore",
    message=".*min_pixels.*max_pixels.*",
    category=FutureWarning,
)

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

    def _generate(self, messages, max_new_tokens: int = 512, **gen_kwargs) -> str:
        """Génère une réponse à partir de messages au format chat Qwen-VL (avec ou sans images)."""
        t0 = time.time()
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)  # (None, None) si pas d'images

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt"
        ).to(self.device)
        print(f"   Tokens d'entrée : {inputs.input_ids.shape[1]}")

        if self.device == "cuda":
            torch.cuda.empty_cache()
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **gen_kwargs)
        t1=time.time()
        print(f'{t1-t0} secondes pour l inférence')
        # Décodage de la sortie (en ignorant les tokens d'entrée)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        del inputs, generated_ids
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return output_text

    def answer_query(self, query: str, k: int = 2) -> tuple[str, str]:
        """
        Retourne deux réponses pour pouvoir les comparer :
          - rag_output      : réponse du VLM avec les k pages récupérées par ColPali
          - baseline_output : réponse du même VLM avec la seule requête (sans RAG)
        """
        print(f"\n🔍 Requête : '{query}'")
        # load mapping
        with open("evaluation/mapping_id_name.json", "r", encoding="utf-8") as f:
            doc_id_to_name = json.load(f)

        # Étape A : Retrieval (Recherche des pages .jpg pertinentes)
        results = self.retriever.search(query, k=k)

        print(f"Pages extraites par ColPali (Top-{k}) :")
        retrieved_pages = [doc_id_to_name.get(str(res.doc_id)) for res in results if res.score > self.th]
        self.last_retrieved_pages = [p for p in retrieved_pages if p]  # pour le benchmark
        retrieved_images = [Image.open(f"data/{doc_name}.jpg") for doc_name in retrieved_pages if doc_name]
        for doc_name in retrieved_pages:
            print(f"   Nom du fichier : {doc_name}")

        # Étape B : Génération AVEC RAG (images + requête)
        if retrieved_images:
            rag_messages = [
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
            print(" Génération de la réponse AVEC RAG...")
            rag_output = self._generate(rag_messages)
        else:
            rag_output = "Aucun document pertinent n'a été trouvé pour répondre à cette question."

        # Étape C : Génération SANS RAG (requête seule, même consigne hors mention des images)
        baseline_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Answer the following question in detail and in technical terms: {query}"
                    }
                ]
            }
        ]
        print(" Génération de la réponse SANS RAG (requête seule)...")
        baseline_output = self._generate(baseline_messages)

        return rag_output, baseline_output

    def judge_closest(self, question: str, ground_truth: str, rag_output: str,
                      baseline_output: str, rag_first: bool = True) -> dict:
        """
        Demande au LLM lequel des deux outputs est le plus proche de la ground truth.
        L'ordre de présentation (A/B) est paramétrable pour compenser le biais de position.
        Retourne {"winner": "rag" | "baseline" | "unparsed", "raw": <sortie brute du juge>}.
        """
        if rag_first:
            answer_a, answer_b, a_is = rag_output, baseline_output, "rag"
        else:
            answer_a, answer_b, a_is = baseline_output, rag_output, "baseline"
        b_is = "baseline" if a_is == "rag" else "rag"

        prompt = (
            "You are an impartial evaluator. Below are a question, a reference answer (ground truth) "
            "and two candidate answers, A and B. Decide which candidate is the closest to the reference "
            "answer in terms of factual content and meaning.\n\n"
            f"Question: {question}\n\n"
            f"Reference answer: {ground_truth}\n\n"
            f"Answer A: {answer_a}\n\n"
            f"Answer B: {answer_b}\n\n"
            "Which answer (A or B) is the closest to the reference answer? Reply with only the letter A or B."
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        raw = self._generate(messages, max_new_tokens=5, do_sample=False)

        match = re.search(r"\b([AB])\b", raw)
        if match is None:
            winner = "unparsed"
        else:
            winner = a_is if match.group(1) == "A" else b_is
        return {"winner": winner, "raw": raw}


def run_benchmark(rag_system: MultimodalRAGInference,
                  benchmark_path: str = "evaluation/benchmark.json",
                  output_path: str = "evaluation/benchmark_results.json",
                  k: int = 2,
                  embed_model_id: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> dict:
    """
    Pose toutes les questions de benchmark.json (clés 'question' et 'answer', ou 'anwer')
    et enregistre pour chacune : réponses avec/sans RAG, cosine similarity avec la ground truth
    et verdict du LLM juge. Les résultats sont sauvegardés au fur et à mesure dans output_path.
    """
    from sentence_transformers import SentenceTransformer  
    import numpy as np

    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    # Embeddings sur CPU pour ne pas consommer la VRAM du VLM
    embedder = SentenceTransformer(embed_model_id, device="cpu")
    embedder.max_seq_length = 512

    per_question = {}
    for i, item in enumerate(benchmark):
        question = item["question"]
        ground_truth = item["answer"]
        print(f"\n=== Question {i + 1}/{len(benchmark)} ===")

        try:
          
            rag_output, baseline_output = rag_system.answer_query(question, k=k)
           
            retrieved_pages = list(rag_system.last_retrieved_pages)

            # Cosine similarity avec la ground truth (embeddings normalisés -> produit scalaire)
            emb = embedder.encode(
                [ground_truth, rag_output, baseline_output],
                normalize_embeddings=True, convert_to_numpy=True,
            )
            cos_rag = float(np.dot(emb[0], emb[1]))
            cos_baseline = float(np.dot(emb[0], emb[2]))

            # Juge LLM : on alterne l'ordre A/B d'une question à l'autre (biais de position)
            verdict = rag_system.judge_closest(
                question, ground_truth, rag_output, baseline_output, rag_first=(i % 2 == 0)
            )

            per_question[str(i)] = {
                "question": question,
                "ground_truth": ground_truth,
                "retrieved_pages": retrieved_pages,
                "rag_answer": rag_output,
                "baseline_answer": baseline_output,
                "cosine_rag": cos_rag,
                "cosine_baseline": cos_baseline,
                "judge_closest": verdict["winner"],
                "judge_raw": verdict["raw"],
            }
        except Exception as e:  # une question en échec (ex. OOM) ne doit pas arrêter tout le benchmark
            print(f"   ⚠️ Erreur sur la question {i} : {e}")
            per_question[str(i)] = {"question": question, "ground_truth": ground_truth, "error": str(e)}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Sauvegarde incrémentale
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"per_question": per_question}, f, indent=2, ensure_ascii=False)

    ok = [r for r in per_question.values() if "error" not in r]
    summary = {
        "n_questions": len(benchmark),
        "n_success": len(ok),
        "n_errors": len(per_question) - len(ok),
        "mean_cosine_rag": float(np.mean([r["cosine_rag"] for r in ok])) if ok else None,
        "mean_cosine_baseline": float(np.mean([r["cosine_baseline"] for r in ok])) if ok else None,
        "judge_wins_rag": sum(r["judge_closest"] == "rag" for r in ok),
        "judge_wins_baseline": sum(r["judge_closest"] == "baseline" for r in ok),
        "judge_unparsed": sum(r["judge_closest"] == "unparsed" for r in ok),
        "n_without_retrieved_pages": sum(len(r["retrieved_pages"]) == 0 for r in ok),
    }
    results = {"summary": summary, "per_question": per_question}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


if __name__ == "__main__":
    rag_system = MultimodalRAGInference(index_name="colpali_index")

    results = run_benchmark(
        rag_system,
        benchmark_path="evaluation/benchmark.json",
        output_path="evaluation/benchmark_results_k3.json",
        k=3,  # k=2 pour limiter les tokens d'entrée
    )
  
    print("\n===== Résumé =====")
    print(json.dumps(results["summary"], indent=2, ensure_ascii=False))