import gzip
import json
import os

def load_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)

doc_id_to_file_name = load_json_gz("C:\\Users\\User\\Documents\\projets_persos_polimi\\Document_retrieval\\.byaldi\\colpali_index\\doc_ids_to_file_names.json.gz")

doc_id_filename = {int(doc_id) :os.path.splitext(os.path.basename(path))[0] for doc_id,path in doc_id_to_file_name.items()}
print(doc_id_filename)

with open("mapping_id_name.json", "w", encoding="utf-8") as f:
    json.dump(doc_id_filename, f, ensure_ascii=False, indent=2)