# -*- coding: utf-8 -*-
"""
➜  images tree -d -L 2 .
.
├── training
│   ├── a
│   ...
│   └── z
└── validation
    ├── a
    ...
    └── z
➜  images tree -d training/a
training/a
├── abbey
├── access_road
├── air_base
├── airfield
├── airlock
├── airplane_cabin
├── airport
│   ├── airport
│   └── entrance
├── airport_terminal
├── airport_ticket_counter
...
"""

import os
import glob
'''
root_path='../data/ADEChallengeData2016/'
splits=['train','val']
folders=['training','validation']

for split,folder in zip(splits,folders):
    glob_images=glob.glob(os.path.join(root_path,'images',folder,'**','*.jpg'),recursive=True)
    glob_annotations=glob.glob(os.path.join(root_path,'annotations',folder,'**','*.png'),recursive=True)
    glob_images.sort()
    glob_annotations.sort()
    print('%s glob images'%split,len(glob_images))
    print('%s glob annotations'%split,len(glob_annotations))
    assert len(glob_images)==len(glob_annotations),'image number %d != annotations number %d'%(len(glob_images),len(glob_annotations))

    write_file=open('./dataset/list/ade20k/'+split+'.txt','w')
    for g_img,g_ann in zip(glob_images,glob_annotations):
        #img_p: eg training/a/abbey/ADE_train_00000981.jpg
        #ann_p: eg training/a/abbey/ADE_train_00000981_seg.png
        img_p=g_img.replace(root_path,'')
        ann_p=g_ann.replace(root_path,'')
        img_basename=os.path.basename(img_p)
        ann_basename=os.path.basename(ann_p)
        assert img_basename.replace('.jpg','.png')==ann_basename,'%s not correpond to %s'%(img_p,ann_p)
        
        write_file.write(img_p+' '+ann_p+'\n')
    write_file.close()
'''

root_path='../data/cocostuff/'
splits=['train','val']
folders=['train2017','val2017']

for split,folder in zip(splits,folders):
    glob_images=glob.glob(os.path.join(root_path,'images',folder,'**','*.jpg'),recursive=True)
    glob_annotations=glob.glob(os.path.join(root_path,'annotations',folder,'**','*.png'),recursive=True)
    glob_images.sort()
    glob_annotations.sort()
    print('%s glob images'%split,len(glob_images))
    print('%s glob annotations'%split,len(glob_annotations))
    assert len(glob_images)==len(glob_annotations),'image number %d != annotations number %d'%(len(glob_images),len(glob_annotations))

    write_file=open('./dataset/list/coco/'+split+'.txt','w')
    for g_img,g_ann in zip(glob_images,glob_annotations):
        #img_p: eg training/a/abbey/ADE_train_00000981.jpg
        #ann_p: eg training/a/abbey/ADE_train_00000981_seg.png
        img_p=g_img.replace(root_path,'')
        ann_p=g_ann.replace(root_path,'')
        img_basename=os.path.basename(img_p)
        ann_basename=os.path.basename(ann_p)
        assert img_basename.replace('.jpg','.png')==ann_basename,'%s not correpond to %s'%(img_p,ann_p)
        
        write_file.write(img_p+' '+ann_p+'\n')
    write_file.close()