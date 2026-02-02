import json
import os

notebook_path = '/Users/apple/Desktop/hira/h-vgxe/vgg16.ipynb'
new_code = r"""# Set dataset path for Autism emotion recogition dataset
base_dir = 'dataset2/Autism emotion recogition dataset/Autism emotion recogition dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

# Ensure paths end with '/' for compatibility
if not train_dir.endswith('/'):
    train_dir += '/'
if not test_dir.endswith('/'):
    test_dir += '/'

row, col = 224, 224
classes = 6  # 6 emotion classes

print("🔄 Using Autism emotion recogition dataset")
print(f"📁 Training directory: {train_dir}")
print(f"📁 Test directory: {test_dir}")
print(f"📐 Image size: {row}x{col}")
print(f"🎯 Number of classes: {classes}")
print("📊 Emotion classes: anger, fear, joy, Natural, sadness, surprise")

def count_exp(path, set_):
    '''Count images in each emotion category'''
    dict_ = {}
    total = 0
    if not os.path.exists(path):
        print(f"⚠️  Warning: Path does not exist: {path}")
        return pd.DataFrame(), 0
    
    for expression in os.listdir(path):
        dir_ = os.path.join(path, expression)
        if os.path.isdir(dir_):
            # Count image files
            count = len([f for f in os.listdir(dir_) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            dict_[expression] = count
            total += count
        else:
            dict_[expression] = 0
    
    df = pd.DataFrame(dict_, index=[set_])
    return df, total

print("\n" + "="*70)
print("DATASET ANALYSIS - AUTISM EMOTION RECOGNITION")
print("="*70)

train_count, train_total = count_exp(train_dir, 'Train')
test_count, test_total = count_exp(test_dir, 'Test')

print("\n📈 Training Set Distribution:")
print(train_count)
print(f"   Total training images: {train_total}")

print("\n📊 Test Set Distribution:")
print(test_count)
print(f"   Total test images: {test_total}")

# Calculate class imbalance
if not train_count.empty and train_total > 0:
    train_dict = train_count.iloc[0].to_dict()
    max_class = max(train_dict, key=train_dict.get)
    min_class = min(train_dict, key=train_dict.get)
    if train_dict[min_class] > 0:
        imbalance_ratio = train_dict[max_class] / train_dict[min_class]
        print(f"\n⚖️  Class imbalance ratio: {imbalance_ratio:.2f}:1 ({max_class} vs {min_class})")

print("="*70)
print("✅ Dataset loaded successfully!")
print("="*70)
"""

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The cell to replace is the one containing "Autistic Children Emotions - Dr. Fatma M. Talaat"
target_string = "Autistic Children Emotions - Dr. Fatma M. Talaat"
found = False

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if target_string in source:
            # Prepare new source lines (notebook source is a list of lines)
            new_source_lines = [line + '\n' for line in new_code.split('\n')]
            # Remove last newline if added excessively
            if new_source_lines[-1] == '\n':
                new_source_lines.pop()
            
            cell['source'] = new_source_lines
            # Clear outputs as they will be stale
            cell['outputs'] = []
            cell['execution_count'] = None
            found = True
            break

if found:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook updated successfully.")
else:
    print("Target cell not found.")
