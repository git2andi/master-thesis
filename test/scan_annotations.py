import os
import xml.etree.ElementTree as ET

annotations_folder = '001-001_annotations'
output_file = 'annotated_frames.txt'

with open(output_file, 'w') as out:
    for xml_file in sorted(os.listdir(annotations_folder)):
        if not xml_file.endswith('.xml'):
            continue

        xml_path = os.path.join(annotations_folder, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        if root.find('object') is not None:
            frame_name = xml_file.replace('.xml', '.jpg')
            out.write(f"{frame_name}\n")

print(f"Annotated frame list saved to {output_file}")
