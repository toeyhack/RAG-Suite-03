# พารามิเตอร์ที่จำเป็นสำหรับ Azure SSO
# .env

TENANT_ID="<ของคุณ-tenant-id>"
CLIENT_ID="<ของคุณ-client-id>"

# ส่วนนี้จะถูกใช้เป็น Secret Key สำรอง หากต้องการสร้าง JWT ภายในแอปพลิเคชัน
# แต่สำหรับการตรวจสอบ Azure Token โดยตรง จะไม่ได้ใช้ค่านี้
# สามารถตั้งค่าเป็นอะไรก็ได้ที่ซับซ้อน หรือปล่อยค่าเริ่มต้นไว้ได้
SECRET_KEY="your-super-secret-key-for-jwt"

# ตั้งค่าสำหรับระบบ RAG
# (ค่าที่เหลือสามารถคงค่าเดิมหรือปรับเปลี่ยนได้ตามต้องการ)
CHROMA_HOST="10.10.32.78"
CHROMA_PORT="8001"
CHROMA_COLLECTION_NAME="rag_documents"
OPENAI_API_KEY=""
