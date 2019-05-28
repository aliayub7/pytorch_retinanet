# coding: utf-8

import os
import sys
import json
from xml.etree import ElementTree
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement


DIR = 'xmls'
WIDTH = 640
HEIGHT = 480
DEPTH = 3


def writeXML(element):
    """
    Write all annotation for one photo
    """
    name = element['External ID']
    xmlData = open(os.path.join(DIR, name[:-4] + '.xml'), 'w')
    annotation = Element('annotation')
    
    folder = SubElement(annotation, 'folder')
    folder.text = '.'
    filename = SubElement(annotation, 'filename')
    filename.text = name
    path = SubElement(annotation, 'path')
    path.text = './images/' + name 
    source = SubElement(annotation, 'source')
    database = SubElement(source, 'database')
    database.text = 'Unknown'
    size = SubElement(annotation, 'size')
    width = SubElement(size, 'width')
    width.text = str(WIDTH)
    height = SubElement(size, 'height')
    height.text = str(HEIGHT)
    depth = SubElement(size, 'depth')
    depth.text = str(DEPTH)
    segmented = SubElement(annotation, 'segmented')
    segmented.text = '0'
    writeObjects(element['Label'], annotation)
    
    xmlData.write(prettify(annotation))


def writeObjects(objects, annotation):
    """
    Write different objects into annotation
    """
    for obj in objects:
        writeLabels(obj, objects[obj], annotation)


def writeLabels(objName, labels, annotation):
    """
    Write different labels of the same object into annotation
    """
    for label in labels:
        obj = SubElement(annotation, 'object')
        name = SubElement(obj, 'name')
        name.text = objName
        pose = SubElement(obj, 'pose')
        pose.text = 'Unspecified'
        truncated = SubElement(obj, 'truncated')
        truncated.text = '0'
        difficult = SubElement(obj, 'difficult')
        difficult.text = '0'
        writeBBox(label['geometry'], obj)


def writeBBox(coor, obj):
    """
    Write one bounding box for one label of one object into annotation
    """
    bndbox = SubElement(obj, 'bndbox')
    xmin = SubElement(bndbox, 'xmin')
    xmin.text = str(coor[0]['x'])
    ymin = SubElement(bndbox, 'ymin')
    ymin.text = str(coor[0]['y'])
    xmax = SubElement(bndbox, 'xmax')
    xmax.text = str(coor[2]['x'])
    ymax = SubElement(bndbox, 'ymax')
    ymax.text = str(coor[2]['y'])


def prettify(elem):
    """
    Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="   ")


if __name__=='__main__':
    if len(sys.argv) != 2:
        print('Please provide .json file')
        os._exit(1)
    path = sys.argv[1] # get json file as a command line argument
    
    if not os.path.isdir(DIR):
        os.makedirs(DIR)
    with open('{}.json'.format(path), 'r') as jsonFile:
        jsonData = json.load(jsonFile)
        for element in jsonData:
            writeXML(element)
    print('Finished converting!')
