import xml.etree.ElementTree as ET
import os 
import pandas as pd

csv_folder = "./csv"
xml_train = "./xml/train"
xml_test = "./xml/test"

def xml_to_csv(path, save_in, name):
    
    xml_list = []
    for xml_file in os.listdir(path):
        tree = ET.parse(os.path.join(path,xml_file))
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df.to_csv(os.path.join(save_in, name), index=False)


if __name__=="__main__":
    xml_to_csv(xml_train, csv_folder, "train.csv")
    xml_to_csv(xml_test, csv_folder, "test.csv")

