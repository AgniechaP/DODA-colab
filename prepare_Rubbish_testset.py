import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm

from utils.utils import read_label, get_layout_image

#Path of Global Wheat Head Detection Dataset
gwhd_2021_path = '/content/drive/MyDrive/DODA/v2/datasets/uavvaste_vol2_resized/'



#Path to save the images of Terraref domain
Terraref_path = '/content/drive/MyDrive/DODA/v2/datasets/uavvaste_vol2_resized/Terraref_vol3/target/'
# makedir(Terraref_path)
Terraref_ori_source_path = '/content/drive/MyDrive/DODA/v2/datasets/uavvaste_vol2_resized/Terraref_vol3/source/'
# makedir(Terraref_ori_source_path)
Terraref_cut_Path = '/content/drive/MyDrive/DODA/v2/datasets/uavvaste_vol2_resized/Terraref_x9_vol3/'
# makedir(Terraref_cut_Path)
Terraref_source_Path = Terraref_cut_Path + 'source/'
# makedir(Terraref_source_Path)
Terraref_target_Path = Terraref_cut_Path + 'target/'
# makedir(Terraref_target_Path)


gwhd_2021_img_path = gwhd_2021_path + 'images/'
csvFiles = gwhd_2021_path + 'test_competition.csv'
Terraref_label = open(Terraref_cut_Path + 'label_x9.txt', 'w', encoding='utf-8')
names_img_with_wheat = open(Terraref_cut_Path + 'with_rubbish.txt', 'w', encoding='utf-8')
names_img_wo_wheat = open(Terraref_cut_Path + 'wo_rubbish.txt', 'w', encoding='utf-8')

#read labels of testset 
label_dic = read_label(csvFiles)


#devide images belonging to Terraref
for img_name in os.listdir(gwhd_2021_img_path):
    if img_name in label_dic.keys():
        domain = label_dic[img_name][-1]
        if 'rubbish' in domain:
            shutil.copy2(gwhd_2021_img_path + img_name, Terraref_path)



for imgName in tqdm(os.listdir(Terraref_path)):
    if '.png' not in imgName:
        continue
    img = cv2.imread(Terraref_path + imgName)

    
    if imgName in label_dic.keys() :
        laBel = label_dic[imgName][0]
        print("laBel: ", laBel)

        imgName = imgName[:-4]
        laBel = laBel.replace(" ", ",").split(';')
        source_img, n_lays = get_layout_image(img, laBel)
        cv2.imwrite(Terraref_ori_source_path + imgName + '.png', source_img)

        if 'no_box' not in laBel:
            # orginal_bbox = np.array([np.array(list(map(int,bbox.split(',')))) for bbox in laBel])    
            orginal_bbox = np.array([np.array(list(map(float, bbox.split(',')))).astype(int) for bbox in laBel])
            # print("original_bbox: ", orginal_bbox)

        #cut image into 512*512, with step=256(x9), and get layout images and new labels(txt)
        i = 0
        for y_i in range(3):
            for x_i in range(3):
                x_start = x_i*256
                y_start = y_i*256
                
                img_cut = img[y_start:y_start+512, x_start:x_start+512,:]
                source_img_cut = source_img[y_start:y_start+512, x_start:x_start+512,:]

                cv2.imwrite(Terraref_target_Path + imgName + '_' + str(i) + '.png', img_cut)
                cv2.imwrite(Terraref_source_Path + imgName + '_' + str(i) + '.png', source_img_cut)

                Terraref_label.write('%s.png'%(Terraref_target_Path + imgName + '_' + str(i)))
                
                if 'no_box' not in laBel:
                    bbox=orginal_bbox.copy()
                    # bbox[:, 0] = bbox[:, 0] - x_start
                    # print("bbox[:, 0]: ", bbox[:, 0])
                    # bbox[:, 2] = bbox[:, 2] - x_start
                    # print("bbox[:, 2]: ", bbox[:, 2])
                    # bbox[:, 1] = bbox[:, 1] - y_start
                    # print("bbox[:, 1] ", bbox[:, 2])
                    # bbox[:, 3] = bbox[:, 3] - y_start
                    # print("bbox[:, 3]: ", bbox[:, 3])

                    # bbox[:, 0:2][bbox[:, 0:2]<0] = 0
                    # bbox[:, 2][bbox[:, 2]>511] = 511
                    # bbox[:, 3][bbox[:, 3]>511] = 511

                    # print("(bbox[:, 2] - bbox[:, 0])", (bbox[:, 2] - bbox[:, 0]))
                    # print("(bbox[:, 3] - bbox[:, 1])", (bbox[:, 3] - bbox[:, 1]))

                    # bboxes = bbox[np.logical_and((bbox[:, 2] - bbox[:, 0])>0, (bbox[:, 3] - bbox[:, 1])>0)] # discard invalid box 

                    bbox[:, 0] = bbox[:, 0] - x_start  # Adjust x coordinate
                    bbox[:, 1] = bbox[:, 1] - y_start  # Adjust y coordinate

                    # Ensure bbox coordinates are within bounds
                    bbox[:, 0] = np.clip(bbox[:, 0], 0, 511)
                    bbox[:, 1] = np.clip(bbox[:, 1], 0, 511)
                    bbox[:, 2] = np.minimum(bbox[:, 2], 511 - bbox[:, 0])  # Adjust width
                    bbox[:, 3] = np.minimum(bbox[:, 3], 511 - bbox[:, 1])  # Adjust height

                    # print("bbox[:, 0]: ", bbox[:, 0])
                    # print("bbox[:, 1] ", bbox[:, 1])
                    print("bbox[:, 2] ", bbox[:, 2])
                    print("bbox[:, 3] ", bbox[:, 3])

                    bboxes = bbox[np.logical_and(bbox[:, 2] > 5, bbox[:, 3] > 5)]  # discard invalid box
                    # print("bboxes: ", bboxes)
                    if bboxes.shape[0] > 0:
                        # print("bboxes.shape > 0 ", bboxes.shape[0])
                        names_img_with_wheat.write('%s.png'%(imgName + '_' + str(i)) + '\n')
                        for bbox in bboxes:
                            bbox = ','.join(map(str, bbox))
                            # print("bbox = ,", bbox)
                            Terraref_label.write(' ' + bbox + ',0')
                    else:
                        names_img_wo_wheat.write('%s.png'%(imgName + '_' + str(i)) + '\n')
                else:
                    names_img_wo_wheat.write('%s.png'%(imgName + '_' + str(i)) + '\n')

                Terraref_label.write('\n')

                

                i += 1
