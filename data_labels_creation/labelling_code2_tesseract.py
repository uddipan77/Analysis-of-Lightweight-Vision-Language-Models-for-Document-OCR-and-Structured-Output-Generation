import os
import base64
import json
import time
import re
import glob
from tqdm import tqdm
import dotenv
import shutil

# --- Configure Tesseract OCR on Windows ---
import pytesseract
# Either set TESSERACT_CMD or modify the default path below
tesseract_cmd = os.environ.get(
    "TESSERACT_CMD",
    r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
)
if not os.path.isfile(tesseract_cmd):
    raise RuntimeError(
        f"Tesseract executable not found at {tesseract_cmd}. "
        "Please install Tesseract OCR and/or set TESSERACT_CMD to its path."
    )
# Point pytesseract at the binary
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

from PIL import Image
from langchain_groq import ChatGroq
from langchain_core.caches import BaseCache
from langchain_core.callbacks.manager import Callbacks
from langchain.schema.messages import HumanMessage, SystemMessage, AIMessage

# Load .env
dotenv.load_dotenv()

# Helper: encode image to base64 for LLM
def image_file_to_base64(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    mime = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png'
    }.get(ext, 'application/octet-stream')
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"

# 1) HTR: first-pass with Tesseract (Fraktur + modern German)
def htr_with_tesseract(image_path: str) -> str:
    img = Image.open(image_path)
    try:
        return pytesseract.image_to_string(img, lang='deu_frak+deu').strip()
    except pytesseract.TesseractError as e:
        raise RuntimeError("Tesseract OCR failed: " + str(e))

# 2) Sort files numerically
def get_sorted_image_files(directory: str, pattern: str = "inventarbuch-*.jpg"):
    paths = glob.glob(os.path.join(directory, pattern))
    def num_key(p):
        m = re.search(r'inventarbuch-(\d+)', p)
        return int(m.group(1)) if m else 0
    return sorted(paths, key=num_key)

# 3) Initialize LLM client
ChatGroq.model_rebuild()  # rebuild internal Pydantic schemas
client = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0
)

# 4) Example images for few-shot learning
example_images = [
    r"C:\\FAU\\sem6\\Thesis\\inventarkarten-ocr\\inventarkarten\\erlangen\\inventarbuch-001.jpg",
    r"C:\\FAU\\sem6\\Thesis\\inventarkarten-ocr\\inventarkarten\\erlangen\\inventarbuch-054.jpg",
    r"C:\\FAU\\sem6\\Thesis\\inventarkarten-ocr\\inventarkarten\\erlangen\\inventarbuch-042.jpg",
    r"C:\\FAU\\sem6\\Thesis\\inventarkarten-ocr\\inventarkarten\\erlangen\\inventarbuch-059.jpg"
]

# 5) Example JSON outputs to teach the model
example_outputs = [
   {
    "image_name": "inventarbuch-001.jpg",
    "inventory_number": "J=fr.",
    "heading": "Anthropologisch-prähiftorische Sammlung der K. Universität Erlangen",
    "measurements": {
        "length": "",
        "breadth": "",
        "depth": ""
    },
    "Material": "Hausen, Ofto",
    "Fundort": "",
    "Fundzeit": "",
    "main_text": "Über eine neue Chronologie des mittleren Palaeolithikums im Vereine mit Breuil und meine Ausgrabungen auf La Micoque Franz Rau, Erlangen. 1916"
    },
    {
        "image_name": "inventarbuch-054.jpg",
        "inventory_number": "J=fr.Eo33",
        "heading": "Anthropologisch-prähistorische Sammlung der K. Universität Erlangen",
        "measurements": {
            "length": "4,32",
            "breadth": "2,46",
            "depth": "1,15"
        },
        "Material": "Primitiver Schaber mit einkerbungen Überarbeitet Eolith/Archaeolith Rutot",
        "Fundort": "Ressain, Vallee dela Laini Belgien",
        "Fundzeit": "",
        "main_text": "Länglich quergelormiger Keil. die Spitze derlinge ist etwa nach der einen Seite abgeplattet. Übrigensber andert der Stein mit seiner dunkel-darckbraunen Oberfläche. Die eine Mansfläche ist glatt leicht scheibenförmig gebaucht, an der der ganzen Fläche einer nicht mehr sichtbar quergelormige Bruchrante. Die andren Fläche zeigt auch eine Britche nicht eine platt concav nach der die fragmente zu von hier auf die Fläche. Hier liegt eine dunnte auf die gerad verlaufene eine Gekrümtheke der Knie, liegen demnächst ganz nicht in die Mitte desders vertaufe. Vielmehr von dies als Mittelkante liegenge"
    },
    # Example for the second image
    {
        "image_name": "inventarbuch-042.jpg",
        "inventory_number": "J=fr.Eo27",
        "heading": "Anthropologisch-prähistorische Sammlung der K. Universität Erlangen",
        "measurements": {
            "length": "7,0",
            "breadth": "5,8",
            "depth": "0,45"
        },
        "Material": "Splitterstück mit Schneiden Eolith/Archaeolith Rutot",
        "Fundort": "Soy, Vallee de la Haine Belgien Artefacte Holin",
        "Fundzeit": "",
        "main_text": "Flacher Stein von achmal herzförmiger Gestalt abgerundeter Spitze. Die eine Fläche ist glatt zeigt nur die spitze eine feine Schlagmarke an derigen nur längsgerissen. Die andre Fläche zeigt am Breiten d ein quer angeordnete schwärzliche Fläche von gelblich hellgrauer Farbe da die der Stein nach Neand beizt. Im Nüben gan für sind auf der Fläche quergelbliche zu den Seiten oder den auffallen ell L'ngöföchen welch durch eins die 'Mbere sind mit verlaffen die Kante gerickeln sind Die Spitze ist oben eine quere Fläche aber stumpfe. Die durch Sticheloje pagenkriften"
    },
    {
        "image_name": "inventarbuch-059.jpg",
        "inventory_number": "",
        "heading": "",
        "measurements": {
            "length": "",
            "breadth": "",
            "depth": ""
        },
        "Material": "",
        "Fundort": "",
        "Fundzeit": "",
        "main_text": "Abschlag enthalt an dem Facetten zeigen das gelbliche Gestein an die sich entsprechenden Fläche. Das ichn vorne vorden ladie des Heims abschlie mit einer kleinen quer anstichlende Flächen das britter anderen Ende mit einer ebensolchen schräg gestellten ab. Beide Flächen sind dünkel durch Abschlag hergestellt. Von einen beiden Seitenenden ist ebenso gerade verlaufende zum Schaben geignet, bei dem andern trifft dies in geringem Grade zu, weil er brübig und etwas zackig ist Von Rutot dem Stolypien zugeeignet."
    }
]

# 6) System message for hybrid approach
system_message = SystemMessage(
    content=(
        "You are an expert OCR assistant specialized in historical German documents.\n"
        "Your task is to extract text from provided images and structure the content into a specific JSON format.\n"
        "The documents are inventory cards from a historical anthropological collection.\n"
        "Return ONLY the valid JSON result with these fields:\n"
        "- image_name: filename of the image\n"
        "- inventory_number: the inventory identifier (e.g., J=fr.Eo33)\n"
        "- heading: document heading/title\n"
        "- measurements: object containing:\n"
        "  - length: measurement in cm (extracted from C-prefixed values)\n"
        "  - breadth: measurement in cm (extracted from B-prefixed values)\n"
        "  - depth: measurement in cm (extracted from D-prefixed values)\n"
        "- Material: description of the material\n"
        "- Fundort: location where the item was found\n"
        "- Fundzeit: time period when found (if available)\n"
        "- main_text: descriptive text about the item\n"
        "Do NOT include any explanation, thinking steps, or commentary in your response. ONLY the JSON object."
    )
)

# 7) Function for hybrid approach with enforced image_name
def process_image_hybrid(image_path: str) -> dict:
    image_name = os.path.basename(image_path)
    ocr_text = htr_with_tesseract(image_path)
    
    # Build few-shot examples
    few_shot = []
    for idx, example_img in enumerate(example_images):
        few_shot.append(
            HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": image_file_to_base64(example_img)}},
                {"type": "text", "text": f"Raw OCR: {htr_with_tesseract(example_img)}"}
            ])
        )
        few_shot.append(AIMessage(content=json.dumps(example_outputs[idx], indent=2)))
    
    # Construct conversation
    conversation = [system_message] + few_shot + [
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": image_file_to_base64(image_path)}},
            {"type": "text", "text": f"Raw OCR: {ocr_text}\n\nProvide only the JSON object without explanations or thinking steps."}
        ])
    ]
    
    # Invoke the model
    response = client.invoke(conversation)
    result_text = response.content.strip()
    
    # Parse JSON with fallback
    result = None
    try:
        result = json.loads(result_text)
    except json.JSONDecodeError:
        m = re.search(r'(\{.*\})', result_text, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group(1))
            except json.JSONDecodeError:
                result = None
    
    if not isinstance(result, dict):
        # Return an error dict if parsing failed
        return {
            "image_name": image_name,
            "error": "Could not parse JSON from LLM response",
            "raw": result_text
        }
    
    # Override to ensure consistency
    result["image_name"] = image_name
    return result

# 8) Main processing function
def main():
    image_dir = r"C:\\FAU\\sem6\\Thesis\\inventarkarten-ocr\\inventarkarten\\erlangen"
    out_dir = os.path.join(image_dir, "ocr_results_hybrid3")
    os.makedirs(out_dir, exist_ok=True)
    
    image_files = get_sorted_image_files(image_dir)
    results = {}
    
    for path in tqdm(image_files, desc="Processing images"):
        fname = os.path.basename(path)
        out_path = os.path.join(out_dir, fname.replace('.jpg', '.json'))
        
        if os.path.exists(out_path):
            print(f"Skipping {fname} - already processed")
            with open(out_path, 'r', encoding='utf-8') as f:
                results[fname] = json.load(f)
            continue
        
        print(f"\nProcessing {fname}...")
        try:
            res = process_image_hybrid(path)
            results[fname] = res
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            print(f"Saved result to {out_path}")
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            results[fname] = {
                "image_name": fname,
                "error": f"Processing error: {e}"
            }
        
        time.sleep(2)
    
    combined_path = os.path.join(out_dir, 'all_results.json')
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nAll results saved to {combined_path}")

if __name__ == '__main__':
    main()
