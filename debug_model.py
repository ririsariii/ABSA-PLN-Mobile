# ============================================================================
# DEBUG SCRIPT - CEK LABEL MODEL
# ============================================================================

import joblib
import json

print("="*80)
print("DEBUGGING MODEL LABELS")
print("="*80)

# Load MultiLabelBinarizer
mlb = joblib.load('models/best_mlb.pkl')

print("\n1. SEMUA LABEL DARI MODEL:")
print("-" * 80)
for i, label in enumerate(mlb.classes_):
    print(f"   {i}: {label}")

print("\n2. JUMLAH LABEL:")
print("-" * 80)
print(f"   Total: {len(mlb.classes_)} labels")

print("\n3. ASPEK UNIK YANG TERDETEKSI:")
print("-" * 80)
aspects = set()
for label in mlb.classes_:
    parts = label.split('_')
    if len(parts) >= 2:
        # Ambil semua bagian kecuali yang terakhir (sentiment)
        aspect = '_'.join(parts[:-1])
        aspects.add(aspect)

for aspect in sorted(aspects):
    aspect_formatted = aspect.replace('_', ' ').title()
    print(f"   - {aspect} → {aspect_formatted}")

print("\n4. REKOMENDASI KODE UNTUK app.py:")
print("-" * 80)
print("   all_aspects = [")
for aspect in sorted(aspects):
    aspect_formatted = aspect.replace('_', ' ').title()
    print(f"       '{aspect_formatted}',")
print("   ]")

print("\n5. LOAD MODEL INFO:")
print("-" * 80)
with open('models/best_model_info.json', 'r') as f:
    model_info = json.load(f)
    
print(f"   Topic Model: {model_info['topic_model']}")
print(f"   Classifier: {model_info['classifier']}")
print(f"   F1-Score: {model_info['metrics']['f1_macro']:.4f}")

print("\n" + "="*80)
print("DEBUGGING SELESAI")
print("="*80)
