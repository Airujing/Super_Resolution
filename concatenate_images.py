import cv2
import os
import numpy as np

def load_images_from_folder(folder, file_extension):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(file_extension):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append((filename, img))
    return images

def annotate_image(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, text, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return image

def resize_images(orig_img, super_res_img):
    # 调整超分图像的大小以匹配原始图像的宽度和高度
    height, width = orig_img.shape[:2]
    super_res_resized = cv2.resize(super_res_img, (width, height))
    return super_res_resized

def concatenate_images(original_folder, super_res_folder, output_folder):
    original_images = load_images_from_folder(original_folder, '.png')[:12]
    super_res_images = load_images_from_folder(super_res_folder, '.bmp')[:12]

    print(f"找到 {len(original_images)} 张原始图像和 {len(super_res_images)} 张超分图像。")

    combined_rows = []

    for orig_name, orig_img in original_images:
        orig_prefix = orig_name[:-4]
        matched = False

        for super_res_name, super_res_img in super_res_images:
            super_res_prefix = super_res_name[:-4]

            if orig_prefix in super_res_prefix:
                matched = True
                
                # 调整超分图像大小
                super_res_img_resized = resize_images(orig_img, super_res_img)

                # 添加标注
                orig_img_annotated = annotate_image(orig_img.copy(), "Original")
                super_res_img_annotated = annotate_image(super_res_img_resized.copy(), "Super Res")
                
                # 将原图和超分图像放入同一行
                combined_row = np.hstack((orig_img_annotated, super_res_img_annotated))
                combined_rows.append(combined_row)
                break
        
        if not matched:
            print(f"未找到匹配的超分图像: {orig_name}")

    # 确保有足够的行以形成 6 行 4 列
    while len(combined_rows) < 6:
        # 创建一个与已存在行相同大小的空白行
        if combined_rows:
            empty_row = np.zeros_like(combined_rows[0])
        else:
            empty_row = np.zeros((500, 2040, 3), dtype=np.uint8)  # 500 是高度，2040 是宽度，可根据需要调整
        combined_rows.append(empty_row)

    # 调整每一行的宽度一致
    max_width = max(row.shape[1] for row in combined_rows)
    for i in range(len(combined_rows)):
        if combined_rows[i].shape[1] < max_width:
            pad_width = max_width - combined_rows[i].shape[1]
            padding = np.zeros((combined_rows[i].shape[0], pad_width, 3), dtype=np.uint8)
            combined_rows[i] = np.hstack((combined_rows[i], padding))

    # 最终拼接成 6 行
    combined_image = np.vstack(combined_rows)
    
    output_path = os.path.join(output_folder, 'combined_image.png')
    cv2.imwrite(output_path, combined_image)
    print(f"生成拼接图像: {output_path}")


# 设置文件夹路径
original_folder = 'DIV2K/DIV2K/DIV2K_test_LR_unknown/X2'  # PNG 文件夹路径
super_res_folder = 'results/swinir_real_sr_x2'  # BMP 文件夹路径
output_folder = 'results'  # 输出文件夹路径

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 执行拼接
concatenate_images(original_folder, super_res_folder, output_folder)
