import os
import random
import glob
import json

def prepare_tlu_states(data_root, target_dir):
    images_dir = os.path.join(data_root, 'images')
    if not os.path.exists(images_dir):
        print(f"Error: {images_dir} does not exist.")
        return

    os.makedirs(target_dir, exist_ok=True)

    class_folders = sorted([f for f in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, f))])
    if len(class_folders) != 23:
        print(f"Warning: Expected 23 classes, found {len(class_folders)}")

    class_to_id = {cls_name: idx for idx, cls_name in enumerate(class_folders)}
    
    with open(os.path.join(target_dir, 'label_map.txt'), 'w', encoding='utf-8') as f:
        for cls_name, idx in class_to_id.items():
            f.write(f"{idx} {cls_name}\n")
    print(f"Saved label mapping for {len(class_to_id)} classes.")

    # Create the metrics JSON for prompting
    metrics = {}
    for cls_name, idx in class_to_id.items():
        # Replace underscores and hyphens with spaces for the prompt
        clean_name = cls_name.replace('_', ' ').replace('-', ' ')
        metrics[str(idx)] = {
            "most_common_name": clean_name,
            "name": clean_name
        }
    
    metrics_path = os.path.join(target_dir, f'tlu_states_metrics-LAION400M.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_path}")

    train_lines = []
    val_lines = []
    test_lines = []
    
    images_by_class = {}

    for cls_name in class_folders:
        cls_id = class_to_id[cls_name]
        cls_dir = os.path.join(images_dir, cls_name)
        
        images = []
        for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG'):
            images.extend(glob.glob(os.path.join(cls_dir, ext)))
        
        images.sort()
        random.seed(42)
        random.shuffle(images)
        images_by_class[cls_id] = images
        
        total_imgs = len(images)
        if total_imgs == 0:
            print(f"Warning: No images found for class {cls_name}")
            continue

        train_split = int(0.7 * total_imgs)
        val_split = int(0.8 * total_imgs)
        
        train_imgs = images[:train_split]
        val_imgs = images[train_split:val_split]
        test_imgs = images[val_split:]
        
        for img in train_imgs:
            train_lines.append(f"{img} {cls_id} 1")
        for img in val_imgs:
            val_lines.append(f"{img} {cls_id} 1")
        for img in test_imgs:
            test_lines.append(f"{img} {cls_id} 1")

    for split_name, lines in [('train.txt', train_lines), ('val.txt', val_lines), ('test.txt', test_lines)]:
        out_path = os.path.join(target_dir, split_name)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"Created {out_path} with {len(lines)} lines.")
        
    # Also create fewshot splits like prepare_fewshot_txt.py does
    for seed in [1, 2, 3]:
        for ct in [1, 2, 4, 8, 16]:
            random.seed(seed)
            fewshot_lines = []
            for cls_id, imgs in images_by_class.items():
                train_pool = imgs[:int(0.7 * len(imgs))] # Use only train split pool
                if len(train_pool) <= ct:
                    sampled = train_pool
                else:
                    sampled = random.sample(train_pool, ct)
                
                for img in sampled:
                    fewshot_lines.append(f"{img} {cls_id} 1")
            
            fs_path = os.path.join(target_dir, f'fewshot{ct}_seed{seed}.txt')
            with open(fs_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(fewshot_lines))
    print("Generated fewshot split files.")
        
    print("Dataset preparation complete.")

if __name__ == '__main__':
    data_root = r"c:\Users\My Computer\Downloads\Dự án TAD-AI-2\tlu-states_2"
    target_dir = r"c:\Users\My Computer\Downloads\Dự án TAD-AI-2\SWAT-GAT\data\tlu_states"
    prepare_tlu_states(data_root, target_dir)
