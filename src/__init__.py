# ทำให้ src เป็น package จริง ๆ (ไม่พึ่ง implicit namespace)
# สำคัญต่อ pickle: คลาส SimplePreprocessor ถูกฝังในโมเดลด้วยชื่อเต็ม
# "src.preprocess.SimplePreprocessor" — ทุก environment ต้อง import ได้ในชื่อนี้
