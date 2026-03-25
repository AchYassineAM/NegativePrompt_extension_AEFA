# scripts/dump_negative_set.py
from config import Negative_SET

print("pnum,stimulus")
for i, s in enumerate(Negative_SET, start=1):
    s_clean = s.replace("\n", "\\n").replace('"', '""')
    print(f'{i},"{s_clean}"')