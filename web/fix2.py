with open(/root/Medical_Project/web/app/dashboard/page.tsx, r) as f:
    text = f.read()
import re
text = re.sub(r\bsetChatHistory\(\[\]\);\s*, ",
