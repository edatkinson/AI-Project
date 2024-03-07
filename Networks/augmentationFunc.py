


def augment_images(directory, null_class_name='Null'):
    for class_name in os.listdir(directory):
        # Skip the null class
        if class_name == null_class_name:
            continue

        class_path = os.path.join(directory, class_name)
        if not os.path.isdir(class_path):
            continue
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path).convert('RGB')
            
            # Apply the transformations
            augmented_img = transform(img)
            
            # Convert back to PIL image to save
            save_img = transforms.ToPILImage()(augmented_img)
            
            # Define a new image name
            base_name, ext = os.path.splitext(img_name)
            new_img_name = f"{base_name}_aug{ext}"
            
            # Save the image back to the same class folder
            save_img.save(os.path.join(class_path, new_img_name))

# using Null as the class which you are excluding during augmentation
#Augments all classes execpt from the Null class, will edit this as we introduce more classes
augment_images('/users/edatkinson/LLL/split_classes/train/', 'Null')
