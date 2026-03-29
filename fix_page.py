import re

with open('web/app/dashboard/page.tsx', 'r') as f:
    text = f.read()

start_str = '<section className="result-section assistant-workbench">'
end_str = '</section>'

start_idx = text.find(start_str)
if start_idx == -1:
    print('Not found')
    exit(1)

end_idx = text.find(end_str, start_idx) + len(end_str)

new_text = text[:start_idx] + '<ClinicalAssistantChat records={analysis.records} />\n          ' + text[end_idx:]

with open('web/app/dashboard/page.tsx', 'w') as f:
    f.write(new_text)
