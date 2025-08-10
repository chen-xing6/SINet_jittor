import os
import shutil
import random

source_dir = '/tmp/pycharm_project_692/Dataset'
target_dir = 'SmallDataset1'


def create_directory_structure():

    os.makedirs(os.path.join(target_dir, 'TrainValDataset', 'Imgs'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'TrainValDataset', 'GT'), exist_ok=True)

    for subdir in ['Imgs', 'GT', 'Edge']:
        os.makedirs(os.path.join(target_dir, 'TestDataset', 'CAMO', subdir), exist_ok=True)

    for subdir in ['Imgs', 'GT', 'Edge']:
        os.makedirs(os.path.join(target_dir, 'TestDataset', 'CHAMELEON', subdir), exist_ok=True)

    for subdir in ['Imgs', 'GT', 'Edge']:
        os.makedirs(os.path.join(target_dir, 'TestDataset', 'COD10K', subdir), exist_ok=True)

    for subdir in ['Imgs', 'GT', 'Instance']:
        os.makedirs(os.path.join(target_dir, 'TestDataset', 'NC4K', subdir), exist_ok=True)


def process_trainval():
    src_imgs = os.path.join(source_dir, 'TrainValDataset', 'Imgs')
    src_gt = os.path.join(source_dir, 'TrainValDataset', 'GT')

    dst_imgs = os.path.join(target_dir, 'TrainValDataset', 'Imgs')
    dst_gt = os.path.join(target_dir, 'TrainValDataset', 'GT')

    all_images = [f for f in os.listdir(src_imgs) if os.path.isfile(os.path.join(src_imgs, f))]
    selected_images = random.sample(all_images, max(1, int(len(all_images) * 0.1)))

    for img in selected_images:
        img_path = os.path.join(src_imgs, img)
        shutil.copy2(img_path, dst_imgs)

        gt_name = os.path.splitext(img)[0] + '.png'
        gt_path = os.path.join(src_gt, gt_name)
        if os.path.exists(gt_path):
            shutil.copy2(gt_path, dst_gt)


def process_cod10k():
    src_imgs = os.path.join(source_dir, 'TestDataset', 'COD10K', 'Imgs')
    src_gt = os.path.join(source_dir, 'TestDataset', 'COD10K', 'GT')
    src_edge = os.path.join(source_dir, 'TestDataset', 'COD10K', 'Edge')

    dst_imgs = os.path.join(target_dir, 'TestDataset', 'COD10K', 'Imgs')
    dst_gt = os.path.join(target_dir, 'TestDataset', 'COD10K', 'GT')
    dst_edge = os.path.join(target_dir, 'TestDataset', 'COD10K', 'Edge')

    category_images = {}
    for img in os.listdir(src_imgs):
        if os.path.isfile(os.path.join(src_imgs, img)):
            parts = img.split('-')
            if len(parts) > 1:
                category = parts[-2]
                category_images.setdefault(category, []).append(img)

    for category, images in category_images.items():
        num_to_select = max(1, int(len(images) * 0.1))
        selected_images = random.sample(images, num_to_select)

        for img in selected_images:
            img_path = os.path.join(src_imgs, img)
            shutil.copy2(img_path, dst_imgs)

            base_name = os.path.splitext(img)[0]
            for ext in ['.png', '.jpg']:
                gt_path = os.path.join(src_gt, base_name + ext)
                if os.path.exists(gt_path):
                    shutil.copy2(gt_path, dst_gt)
                    break

            for ext in ['.png', '.jpg']:
                edge_path = os.path.join(src_edge, base_name + ext)
                if os.path.exists(edge_path):
                    shutil.copy2(edge_path, dst_edge)
                    break


def process_nc4k():
    src_imgs = os.path.join(source_dir, 'TestDataset', 'NC4K', 'Imgs')
    src_gt = os.path.join(source_dir, 'TestDataset', 'NC4K', 'GT')
    src_instance = os.path.join(source_dir, 'TestDataset', 'NC4K', 'Instance')

    dst_imgs = os.path.join(target_dir, 'TestDataset', 'NC4K', 'Imgs')
    dst_gt = os.path.join(target_dir, 'TestDataset', 'NC4K', 'GT')
    dst_instance = os.path.join(target_dir, 'TestDataset', 'NC4K', 'Instance')

    all_images = [f for f in os.listdir(src_imgs) if os.path.isfile(os.path.join(src_imgs, f))]
    selected_images = random.sample(all_images, max(1, int(len(all_images) * 0.1)))

    for img in selected_images:
        img_path = os.path.join(src_imgs, img)
        shutil.copy2(img_path, dst_imgs)

        base_name = os.path.splitext(img)[0]

        for ext in ['.png', '.jpg']:
            gt_path = os.path.join(src_gt, base_name + ext)
            if os.path.exists(gt_path):
                shutil.copy2(gt_path, dst_gt)
                break

        for ext in ['.png', '.jpg', '.txt']:
            instance_path = os.path.join(src_instance, base_name + ext)
            if os.path.exists(instance_path):
                shutil.copy2(instance_path, dst_instance)
                break


def copy_full_directory(src, dst):
    if os.path.exists(src):
        if not os.path.exists(dst):
            shutil.copytree(src, dst)
        else:
            for item in os.listdir(src):
                src_item = os.path.join(src, item)
                dst_item = os.path.join(dst, item)
                if os.path.isdir(src_item):
                    shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_item, dst_item)


def main():
    create_directory_structure()

    print("Processing TrainValDataset...")
    process_trainval()

    print("Processing CAMO...")
    copy_full_directory(
        os.path.join(source_dir, 'TestDataset', 'CAMO'),
        os.path.join(target_dir, 'TestDataset', 'CAMO')
    )

    print("Processing CHAMELEON...")
    copy_full_directory(
        os.path.join(source_dir, 'TestDataset', 'CHAMELEON'),
        os.path.join(target_dir, 'TestDataset', 'CHAMELEON')
    )

    print("Processing COD10K...")
    process_cod10k()

    print("Processing NC4K...")
    process_nc4k()

    print("\nSmall dataset creation completed successfully!")
    print(f"Small dataset saved to: {target_dir}")


if __name__ == "__main__":
    main()