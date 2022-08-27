import glob
import numpy as np
import os
from os.path import join
import cv2
import xml.etree.ElementTree as ET
import sys
from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import Document
from tqdm import tqdm
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

# origin_dir = '原图像存放地址'
# target_dir1 = '分块图像存放地址'
# annota_dir = '原boundingbox的xml文件存放地址'
# target_dir2 = '分块boundingbox的xml文件存放地址'


def clip_img(No, oriname, win_size, stride):
    from_name = os.path.join(origin_dir, oriname+'.jpg')
    img = cv2.imread(from_name)
    h_ori, w_ori, _ = img.shape  # 保存原图的大小
    # img = cv2.resize(img, (5472, 3648))#可以resize也可以不resize，看情况而定
    h, w, _ = img.shape
    xml_name = os.path.join(annota_dir, oriname+'.xml')  # 读取每个原图像的xml文件
    xml_ori = ET.parse(xml_name).getroot()
    res = np.empty((0, 5))  # 存放坐标的四个值和类别
    for obj in xml_ori.iter('object'):
        difficult = int(obj.find('difficult').text) == 1
        if difficult:
            continue
        name = obj.find('name').text.lower().strip()
        bbox = obj.find('bndbox')
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text) - 1
            cur_pt = int(
                cur_pt*h/h_ori) if i % 2 == 1 else int(cur_pt * w / w_ori)
            bndbox.append(cur_pt)
        #label_idx = self.class_to_ind[name]
        bndbox.append(name)
        res = np.vstack((res, bndbox))
    i = 0
    # win_size = 1368  # 分块的大小
    # stride = 684  # 重叠的大小，设置这个可以使分块有重叠
    for r in range(0, h - win_size, stride):
        for c in range(0, w - win_size, stride):
            # flag = np.zeros([1,100]) # 这里不应该只有10吧，不然后面第75行 re对象个数基本会超过10个obj
            flag = np.zeros([1, len(res)])
            youwu = False
            xiefou = True
            tmp = img[r: r+win_size, c: c+win_size]
            for re in range(res.shape[0]):
                xmin, ymin, xmax, ymax, label = res[re]
                if int(xmin) >= c and int(xmax) <= c+win_size and int(ymin) >= r and int(ymax) <= r+win_size:
                    flag[0][re] = 1
                    youwu = True
                elif int(xmin) < c or int(xmax) > c+win_size or int(ymin) < r or int(ymax) > r+win_size:
                    pass
                else:
                    xiefou = False
                    break
            if xiefou:  # 如果物体被分割了，则忽略不写入
                if youwu:  # 有物体则写入xml文件
                    doc = Document()
                    annotation = doc.createElement('annotation')
                    doc.appendChild(annotation)
                    for re in range(res.shape[0]):
                        xmin, ymin, xmax, ymax, label = res[re]
                        xmin = int(xmin)
                        ymin = int(ymin)
                        xmax = int(xmax)
                        ymax = int(ymax)
                        if flag[0][re] == 1:
                            xmin = str(xmin-c)
                            ymin = str(ymin-r)
                            xmax = str(xmax-c)
                            ymax = str(ymax-r)
                            object_charu = doc.createElement('object')
                            annotation.appendChild(object_charu)
                            name_charu = doc.createElement('name')
                            name_charu_text = doc.createTextNode(label)
                            name_charu.appendChild(name_charu_text)
                            object_charu.appendChild(name_charu)
                            dif = doc.createElement('difficult')
                            dif_text = doc.createTextNode('0')
                            dif.appendChild(dif_text)
                            object_charu.appendChild(dif)
                            bndbox = doc.createElement('bndbox')
                            object_charu.appendChild(bndbox)
                            xmin1 = doc.createElement('xmin')
                            xmin_text = doc.createTextNode(xmin)
                            xmin1.appendChild(xmin_text)
                            bndbox.appendChild(xmin1)
                            ymin1 = doc.createElement('ymin')
                            ymin_text = doc.createTextNode(ymin)
                            ymin1.appendChild(ymin_text)
                            bndbox.appendChild(ymin1)
                            xmax1 = doc.createElement('xmax')
                            xmax_text = doc.createTextNode(xmax)
                            xmax1.appendChild(xmax_text)
                            bndbox.appendChild(xmax1)
                            ymax1 = doc.createElement('ymax')
                            ymax_text = doc.createTextNode(ymax)
                            ymax1.appendChild(ymax_text)
                            bndbox.appendChild(ymax1)
                        else:
                            continue
                    xml_name = oriname+'_1368%d.xml' % (i)
                    to_xml_name = os.path.join(target_dir2, xml_name)
                    with open(to_xml_name, 'wb+') as f:
                        f.write(doc.toprettyxml(indent="\t", encoding='utf-8'))
                    #name = '%02d_%02d_%02d_.bmp' % (No, int(r/win_size), int(c/win_size))
                    img_name = oriname+'_1368%d.jpg' % (i)
                    to_name = os.path.join(target_dir1, img_name)
                    i = i+1
                    cv2.imwrite(to_name, tmp)


def getImagesInDir():
    image_list = []
    for ext in ["*.JPG", "*.jpg", "*.png", "*.jpeg"]:
        filenames = glob.glob(os.path.join(target_dir1, ext))
        for filename in filenames:
            image_name = filename.split("\\")[-1]
            image_list.append(image_name)
    return image_list


def convert(size, box):

    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh

    return (x, y, w, h)


def convert_annotation(image_path):
    global win_size
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]

    in_file = open(target_dir2+'\\' + basename_no_ext + '.xml')
    out_file = open(target_dir3 + '\\' + basename_no_ext + '.txt', 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    # size = root.find('size')
    # w = int(size.find('width').text)
    # h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(
            xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((win_size, win_size), b)
        out_file.write(str(cls_id) + " " + str(bb[0]) + " " + str(
            bb[1]) + " " + str(bb[2]) + " " + str(bb[3]) + '\n')


def VOC2YOLO():
    image_paths = getImagesInDir()
    for image_path in image_paths:
        convert_annotation(image_path)
    print("Finished processing")


def unconvert(class_id, width, height, x, y, w, h):
    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)
    class_id = int(class_id)
    return (class_id, xmin, xmax, ymin, ymax)


def YOLO2VOC(classes):

    classes = list(classes)
    ids = list()
    l = os.listdir(labels_path)

    check = '.DS_Store' in l
    if check == True:
        l.remove('.DS_Store')

    ids = [x.split('.')[0] for x in l]

    annopath = join(labels_path, '%s.txt')
    imgpath = join(img_path, '%s.jpg')

    if not os.path.exists(path+'/voc'):
        os.makedirs(path+'/voc')

    outpath = join(path+'/voc', '%s.xml')

    for i in range(len(ids)):
        img_id = ids[i]
        img = cv2.imread(imgpath % img_id)
        height, width, channels = img.shape

        node_root = Element('annotation')
        node_folder = SubElement(node_root, 'folder')
        node_folder.text = 'yh'
        img_name = img_id + '.jpg'

        node_filename = SubElement(node_root, 'filename')
        node_filename.text = img_name

        node_source = SubElement(node_root, 'source')
        node_database = SubElement(node_source, 'database')
        node_database.text = 'Coco database'

        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = str(width)

        node_height = SubElement(node_size, 'height')
        node_height.text = str(height)

        node_depth = SubElement(node_size, 'depth')
        node_depth.text = str(channels)

        node_segmented = SubElement(node_root, 'segmented')
        node_segmented.text = '0'

        target = (annopath % img_id)
        if os.path.exists(target):
            label_norm = np.loadtxt(target).reshape(-1, 5)

            for i in range(len(label_norm)):
                labels_conv = label_norm[i]
                new_label = unconvert(
                    labels_conv[0], width, height, labels_conv[1], labels_conv[2], labels_conv[3], labels_conv[4])
                node_object = SubElement(node_root, 'object')
                node_name = SubElement(node_object, 'name')
                node_name.text = classes[new_label[0]]

                node_pose = SubElement(node_object, 'pose')
                node_pose.text = 'Unspecified'

                node_truncated = SubElement(node_object, 'truncated')
                node_truncated.text = '0'
                node_difficult = SubElement(node_object, 'difficult')
                node_difficult.text = '0'
                node_bndbox = SubElement(node_object, 'bndbox')
                node_xmin = SubElement(node_bndbox, 'xmin')
                node_xmin.text = str(new_label[1])
                node_ymin = SubElement(node_bndbox, 'ymin')
                node_ymin.text = str(new_label[3])
                node_xmax = SubElement(node_bndbox, 'xmax')
                node_xmax.text = str(new_label[2])
                node_ymax = SubElement(node_bndbox, 'ymax')
                node_ymax.text = str(new_label[4])
                xml = tostring(node_root, pretty_print=True)
                dom = parseString(xml)
        f = open(outpath % img_id, "wb")
        f.write(xml)
        f.close()
    print("Finished processing")


if __name__ == '__main__':

    path = os.path.dirname(os.path.realpath(sys.argv[0]))

    files = ['images_crop', 'voc_crop', 'yolo_label_crop']
    for file in files:
        if not os.path.exists(path+'\\'+file):
            os.makedirs(path+'\\'+file)

    with open(path+'/classes.txt', "r") as f:  # 打开文件
        classes = list(f.read().split(','))   # 读取文件
    print("类别："+str(classes))
    img_path = input('原始影像文件夹地址：')
    labels_path = input('原始标注(labels)文件夹地址：')

    YOLO2VOC(classes)

    origin_dir = img_path
    target_dir1 = os.path.join(path, 'images_crop')
    annota_dir = os.path.join(path, 'voc')
    target_dir2 = os.path.join(path, 'voc_crop')
    target_dir3 = os.path.join(path, 'yolo_label_crop')  # 最终标注yolo_label结果

    win_size = int(input('请输入分割像素尺寸(pix)：'))
    stride = int(input('请输入重叠像素尺寸(pix)：'))

    for No, name in tqdm(enumerate(os.listdir(origin_dir))):
        clip_img(No, name[:-4], win_size, stride)

    VOC2YOLO()
