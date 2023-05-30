def apply(self, image, ann):
        # get the keypoints from all the annotations
        keypoints = ann['keypoints']
        
        # Implement the SDA logic here
        # 1. Check if the augmentation should be applied based on the probability
        #if random.random() > self.probability:
        #    return image, anns, meta

        # 2. Perform the augmentation:
        #   - choose random number of body parts to add to the image
        #   - choose random body parts from the pool of body parts
        #   - add them to the image
        
        nb_bodyparts = random.randint(1, NB_BODY_PARTS)
        #print("nb_bodyparts ",nb_bodyparts)
        augmented_image = np.asarray(image, dtype=np.uint8).copy()
        # get the image dimensions
        image_height, image_width = augmented_image.shape[:2]
        # load the body parts pool
        bodyparts = json.load(open(bodypart_file))
        for i in range(nb_bodyparts):
            # choose a random body part from the pool
            bodypart = random.choice(bodyparts)
            print(bodypart)
            # load the body part
            bodypart = plt.imread(bodypart)
            
            # get the body part dimensions
            bodypart_height, bodypart_width = bodypart.shape[:2]
            # ensure the body part is not too big compared to the image
            if image_height/bodypart_height > IMG_TO_BODYPART_RATION or image_width/bodypart_width > IMG_TO_BODYPART_RATION:
                # choose a random position to add the body part
                # ensure image_width - bodypart_width > 0
                # ensure image_height - bodypart_height > 0
                if image_width - bodypart_width > 0 and image_height - bodypart_height > 0:
                    # choose a random position to add the body part not directly on top of keypoints
                    x = random.randint(0, image_width - bodypart_width)
                    y = random.randint(0, image_height - bodypart_height)
                    nb_retries = 0
                    not_on_kp = True
                    # to avoid infinite loop
                    while nb_retries < 5:
                        not_on_kp = True
                        # check if the body part is not on top of keypoints
                        for i in range(0, len(keypoints), 3):
                            if  x < keypoints[i] - KP_DIST_THRESHOLD and \
                                keypoints[i] + KP_DIST_THRESHOLD < x + bodypart_width and \
                                y < keypoints[i + 1] - KP_DIST_THRESHOLD and \
                                keypoints[i + 1] + KP_DIST_THRESHOLD < y + bodypart_height:
                                    continue
                            else:
                                x = random.randint(0, image_width - bodypart_width)
                                y = random.randint(0, image_height - bodypart_height)
                                nb_retries += 1
                                not_on_kp = False
                        if not_on_kp:
                            break
                        
                    # TODO:rotate image ? 
                    # add the body part to the image
                    # if body
                    if not_on_kp == True:
                        
                        augmented_image[y : y+bodypart_height, x : x+bodypart_width] = bodypart
                        augmented_image = draw_keypoint(augmented_image, keypoints)

        # 3. Return the augmented image
            #transform to pil image
        print(not_on_kp)
        augmented_image = Image.fromarray(augmented_image)

        return augmented_image
   