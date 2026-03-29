import re

with open('/root/Medical_Project/web/app/dashboard/page.tsx', 'r') as f:
    text = f.read()

start_str = '<section className=" result-section assistant-workbench\>'
end_str
