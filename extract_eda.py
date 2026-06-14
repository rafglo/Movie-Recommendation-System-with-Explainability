import json
import base64
import os

with open('notebooks/eda.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

os.makedirs('presentation_assets', exist_ok=True)
img_idx = 1

print("--- Markdown Cells ---")
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        print(''.join(cell['source']))
        print('-'*40)
    elif cell['cell_type'] == 'code':
        for output in cell.get('outputs', []):
            if 'data' in output and 'image/png' in output['data']:
                img_data = base64.b64decode(output['data']['image/png'])
                img_path = f'presentation_assets/eda_plot_{img_idx}.png'
                with open(img_path, 'wb') as img_f:
                    img_f.write(img_data)
                print(f"Saved image: {img_path}")
                img_idx += 1
