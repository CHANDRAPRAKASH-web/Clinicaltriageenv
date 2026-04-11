import zipfile
import xml.etree.ElementTree as ET

def extract_text_from_docx(file_path):
    try:
        with zipfile.ZipFile(file_path) as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.fromstring(xml_content)
            
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            text = []
            for node in tree.findall('.//w:t', namespaces):
                if node.text:
                    text.append(node.text)
            
            return '\n'.join(text)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    content = extract_text_from_docx(r"\\wsl.localhost\Ubuntu-20.04\home\chandraprakash\clinicaltriageenv\ClinicalTriageEnv_PRD.docx")
    with open(r"\\wsl.localhost\Ubuntu-20.04\home\chandraprakash\clinicaltriageenv\prd_text.txt", "w", encoding="utf-8") as f:
        f.write(content)
